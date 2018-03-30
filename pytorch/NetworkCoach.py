import torch
import time

import numpy as np

from AccuracyCalculator import AccuracyCalculator

#----------------------------------------------------------------------------------------
#---- This class provides methods for training, testing and evaluating networks.
#---- This wrapper is useful when several networks with different parameters have to be
#---- trained and tested

class NetworkCoach:

    #--------------------------------------------------------------------------------
    #---- This method trains a network
    #---- In: datasetLoaderTrain - dataset loader for training set
    #---- In: datasetLoaderValidate - dataset loader for validation set
    #---- In: nnModel - a neural network model
    #---- In: nnEpochs - number of epochs to train the network
    #---- In: nnLoss - a loss function for training the network
    #---- In: nnOptimizer - optimization method
    #---- In: scheduler - learning rate changed sheduler
    #---- In: pathOutputModel - the best model will be saved here
    #---- In: pathOutputLog - the log of training (loss, time) will be saved here

    def netTrain (datasetLoaderTrain, datasetLoaderValidate, nnModel, nnEpochs, nnLoss, nnOptimizer, scheduler, pathOutputModel, pathOutputLog):

        ostreamLog = open(pathOutputLog, 'a')

        lossMIN = 100000

        for epochID in range(0, nnEpochs):

            NetworkCoach.epochTrain(nnModel, datasetLoaderTrain, nnOptimizer, nnLoss)
            lossVal, losstensor = NetworkCoach.epochVal(nnModel, datasetLoaderValidate, nnOptimizer, nnLoss)

            scheduler.step(losstensor.data[0])

            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampEpochEnd = timestampDate + '-' + timestampTime

            if lossVal < lossMIN:
                lossMIN = lossVal
                torch.save({'epoch': epochID + 1, 'state_dict': nnModel.state_dict(), 'best_loss': lossMIN, 'optimizer': nnOptimizer.state_dict()}, pathOutputModel)
                ostreamLog.write("EPOCH [" + str(epochID) + "]: " + timestampEpochEnd + " loss: " + str(lossVal) + " {SAVED} \n")
                ostreamLog.flush()
            else:
                ostreamLog.write("EPOCH [" + str(epochID) + "]: " + timestampEpochEnd + " loss: " + str(lossVal) + " {-SKP-} \n")
                ostreamLog.flush()

        ostreamLog.close()

    #--------------------------------------------------------------------------------
    #---- Method to test the trained model
    #---- In: datasetLoaderTest - dataset loader for the testing set
    #---- In: nnModel - model (the weights should already be loaded)
    #---- In: pathOutputAcc - the accuracy statistics will be saved here
    #---- In: pathOutputPred - ground truth and predicted vectors will be saved here

    def netTest (datasetLoaderTest, nnModel, nnClassCount, pathOutputAcc, pathOutputPred):

        outGT = torch.FloatTensor().cuda()
        outPRED = torch.FloatTensor().cuda()

        #---- Switch model to the evaluation mode
        nnModel.eval()

        #---- Do testing here
        for i, (input, target) in enumerate(datasetLoaderTest):
            target = target.cuda()
            outGT = torch.cat((outGT, target), 0)

            bs, n_crops, c, h, w = input.size()

            varInput = torch.autograd.Variable(input.view(-1, c, h, w).cuda(), volatile=True)

            out = nnModel(varInput)
            outMean = out.view(bs, n_crops, -1).mean(1)

            outPRED = torch.cat((outPRED, outMean.data), 0)


        dataGT = outGT.cpu().numpy()
        dataPRED = outPRED.cpu().numpy()

        NetworkCoach.saveResultPredictions(dataGT, dataPRED, pathOutputPred)
        NetworkCoach.saveResultAccuracyStatistics(dataGT, dataPRED, pathOutputAcc)

    #--------------------------------------------------------------------------------
    #---- Method to perform one epoch (loop over the whole dataset) of training
    #---- In: model -- network model
    #---- In: dataLoader -- dataset loader
    #---- In: optimizer - optimization method
    #---- In: loss - loss function

    def epochTrain(model, dataLoader, optimizer, loss):

        model.train()

        for batchID, (input, target) in enumerate(dataLoader):
            target = target.cuda(async=True)

            varInput = torch.autograd.Variable(input)
            varTarget = torch.autograd.Variable(target)
            varOutput = model(varInput)

            lossvalue = loss(varOutput, varTarget)

            #---- Back-propagation
            optimizer.zero_grad()
            lossvalue.backward()
            optimizer.step()

    # --------------------------------------------------------------------------------

    def epochVal(model, dataLoader, optimizer, loss):

        model.eval()

        lossVal = 0
        lossValNorm = 0

        losstensorMean = 0

        for i, (input, target) in enumerate(dataLoader):
            target = target.cuda(async=True)

            varInput = torch.autograd.Variable(input, volatile=True)
            varTarget = torch.autograd.Variable(target, volatile=True)
            varOutput = model(varInput)

            losstensor = loss(varOutput, varTarget)
            losstensorMean += losstensor

            lossVal += losstensor.data[0]
            lossValNorm += 1

        outLoss = lossVal / lossValNorm
        losstensorMean = losstensorMean / lossValNorm

        return outLoss, losstensorMean

    # --------------------------------------------------------------------------------
    #---- This method saves the ground truth and predicted data into a file
    #---- One line of the file contains ground truth and then predicted vectors
    #---- In: dataGT - ground truth data (numpy array)
    #---- In: dataPRED - predicted data (numpy array)
    #---- In: pathOutput - file to save the data

    def saveResultPredictions (dataGT, dataPRED, pathOutput):
        ostream = open(pathOutput, 'w')

        length = len(dataGT)

        for i in range(0, length):
            vectorGroundTruth = dataGT[i]
            vectorPredicted = dataPRED[i]

            dim = len(vectorGroundTruth)

            for k in range(0, dim):
                ostream.write(str(vectorGroundTruth[k]) + " ")

            for k in range (0, dim):
                ostream.write(str(vectorPredicted[k]) + " ")

            ostream.write("\n")

        ostream.flush()
        ostream.close()

    def saveResultAccuracyStatistics (dataGT, dataPRED, pathOutput):

        ostream = open(pathOutput, 'w')

        dim = len(dataGT[0])

        aurocIndividual = AccuracyCalculator.computeAUROC(dataGT, dataPRED, dim)
        aurocMean = np.array(aurocIndividual).mean()

        oTP, oFP, oTN, oFN = AccuracyCalculator.computeAccuracyStatistics(dataGT, dataPRED, dim)

        oACC = AccuracyCalculator.computeAccuracyPerClass(oTP, oFP, oTN, oFN)
        oFS = AccuracyCalculator.computeAccuracyFscore(oTP, oFP, oTN, oFN)

        ostream.write("AUROC-MEAN " + str(aurocMean) + "\n")

        ostream.write("AUROC TP FP TN FN ACC FS\n")
        for i in range(0, dim):
            ostream.write(str(aurocIndividual[i]) + " ")
            ostream.write(str(oTP[i]) + " ")
            ostream.write(str(oFP[i]) + " ")
            ostream.write(str(oTN[i]) + " ")
            ostream.write(str(oFN[i]) + " ")
            ostream.write(str(oACC[i]) + " ")
            ostream.write(str(oFS[i]) + " ")
            ostream.write("\n")

        ostream.flush()
        ostream.close()

#----------------------------------------------------------------------------------------