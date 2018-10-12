import torch
import time
import numpy as np
import sys

sys.path.append('../')
from accuracy.ScoreCalculator import ScoreCalculator

class AlgorithmClassification():

    # ------------------------------------ PRIVATE ----------------------------------

    def __epochtrain__ (model, dataLoader, optimizer, loss):
        """
        :param model: network model
        :param dataLoader: data loader for the training set
        :param optimizer: optimizer for the training procedure
        :param loss: loss function
        :return: mean loss for the training dataset for this epoch
        """
        model.train()

        lossMean = 0
        lossMeanNorm = 0

        for batchID, (input, target) in enumerate(dataLoader):
            target = target.cuda()

            output = model(input)

            lossvalue = loss(output, target)

            lossMean += lossvalue.item()
            lossMeanNorm += 1

            # ---- Back-propagation
            optimizer.zero_grad()
            lossvalue.backward()
            optimizer.step()

        return lossMean / lossMeanNorm

    # -------------------------------------------------------------------------------

    def __epochval__ (model, dataLoader, optimizer, loss):
        """
        :param model: network model
        :param dataLoader: data loader for the validation set
        :param optimizer: optimizer for the training procedure
        :param loss: loss function
        :return: mean loss for the validation dataset for this epoch
        """

        model.eval()

        lossVal = 0
        lossValNorm = 0

        losstensorMean = 0

        for i, (input, target) in enumerate(dataLoader):
            target = target.cuda()

            output = model(input)

            losstensor = loss(output, target)
            losstensorMean += losstensor

            # ---- For the pytorch version 0.4.0 direct conversion from 0-dim tensor to float is allowed
            lossVal += losstensor.item()
            lossValNorm += 1

        outLoss = lossVal / lossValNorm
        losstensorMean = losstensorMean / lossValNorm

        return outLoss, losstensorMean

    # -------------------------------------------------------------------------------

    def __saveoutput__(outgt, outpred, path):
        """
        :param outgt: list of ground truth vectors
        :param outpred: list of predicted vectors
        :param path: output file path
        """
        ostream = open(path, 'w')

        length = len(outgt)

        for i in range(0, length):
            vectorGroundTruth = outgt[i]
            vectorPredicted = outpred[i]

            dim = len(vectorGroundTruth)

            for k in range(0, dim):
                ostream.write(str(vectorGroundTruth[k]) + " ")

            for k in range(0, dim):
                ostream.write(str(vectorPredicted[k]) + " ")

            ostream.write("\n")

        ostream.flush()
        ostream.close()

    # -------------------------------------------------------------------------------

    def __savescore__(outgt, outpred, path):
        """
        :param outgt: list of ground truth vectors
        :param outpred: list of predicted vectors
        :param path: output file path
        """
        ostream = open(path, 'w')

        dim = len(outgt[0])


        aurocIndividual = ScoreCalculator.computeAUROC(outgt, outpred, dim)
        aurocMean = np.array(aurocIndividual).mean()

        oTP, oFP, oTN, oFN = ScoreCalculator.computeAccuracyStatistics(outgt, outpred, dim)

        oACC = ScoreCalculator.computeAccuracyPerClass(oTP, oFP, oTN, oFN)
        oFS = ScoreCalculator.computeAccuracyFscore(oTP, oFP, oTN, oFN)

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

    # ------------------------------------ PUBLIC ------------------------------------

    def train (dloaderTrain, dloaderValidate, nnModel, nnEpochs, nnLoss, nnOptimizer, scheduler, pathOutputModel, pathOutputLog = None, isSilentMode = True):
        """
        Train a network
        :param dloaderTrain: dataset loader for the training set
        :param dloaderValidate: dataset loader for the validation set
        :param nnModel: network model
        :param nnEpochs: number of epochs to train
        :param nnLoss: loss function
        :param nnOptimizer: optimizer function
        :param scheduler: scheduler procedure
        :param pathOutputModel: path to save the best model
        :param pathOutputLog: [optional] path to save the loss log file
        :param isSilentMode: [optional] (bool) output log into console or not
        :return:
        """

        if pathOutputLog is not None:
            ostreamLog = open(pathOutputLog, 'a')

        lossMIN = 100000

        for epochID in range(0, nnEpochs):

            lossTrain = AlgorithmClassification.__epochtrain__(nnModel, dloaderTrain, nnOptimizer, nnLoss)

            # ---- Don't save gradients to prevent memory leakage
            with torch.no_grad():
                lossVal, losstensor = AlgorithmClassification.__epochval__(nnModel, dloaderValidate, nnOptimizer, nnLoss)

            # ---- For the pytorch version 0.4.0 direct conversion from 0-dim tensor to float is allowed
            scheduler.step(losstensor)

            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampEpochEnd = timestampDate + '-' + timestampTime

            if lossVal < lossMIN:
                lossMIN = lossVal
                torch.save({'epoch': epochID + 1, 'state_dict': nnModel.state_dict(), 'best_loss': lossMIN, 'optimizer': nnOptimizer.state_dict()}, pathOutputModel)
                if pathOutputLog is not None:
                    ostreamLog.write("\nEPOCH [" + str(epochID) + "]: " + timestampEpochEnd + " loss_tr: " + str(lossTrain)  + " loss_val: " + str(lossVal) + " {SAVED}")
                    ostreamLog.flush()

                if isSilentMode == False:
                    print("EPOCH [" + str(epochID) + "]: " + timestampEpochEnd + " loss_tr: " + str(lossTrain)  + " loss_val: " + str(lossVal) + " {SAVED}")

            else:
                if pathOutputLog is not None:
                    ostreamLog.write("\nEPOCH [" + str(epochID) + "]: " + timestampEpochEnd + " loss_tr: " + str(lossTrain)  + " loss_val: " + str(lossVal) + " {-SKIP-}")
                    ostreamLog.flush()

                if isSilentMode == False:
                    print("EPOCH [" + str(epochID) + "]: " + timestampEpochEnd + " loss_tr: " + str(lossTrain)  + " loss_val: " + str(lossVal) + " {-SKIP-}")

        if pathOutputLog != None:
            ostreamLog.close()

    # -------------------------------------------------------------------------------

    def test (dLoaderTest, nnModel, pathOutputAcc, pathOutputPred, isSilentMode = True):
        """
        :param dLoaderTest: dataset loader for the test set
        :param nnModel: trained network model
        :param nnClassCount: number of classes / dimension of the output vector
        :param pathOutputAcc: path to the file where accuracy scores will be saved
        :param pathOutputPred: path to the file where predicted vectors
        :return:
        """

        outGT = torch.FloatTensor().cuda()
        outPRED = torch.FloatTensor().cuda()

        # ---- Switch model to the evaluation mode
        nnModel.eval()

        # ---- Do testing here
        for i, (input, target) in enumerate(dLoaderTest):

            if isSilentMode == False:
                print("[TESTING]: sample " + str(i) + " " + str(len(dLoaderTest)))

            target = target.cuda()
            outGT = torch.cat((outGT, target), 0)

            bs, n_crops, c, h, w = input.size()

            varInput = torch.autograd.Variable(input.view(-1, c, h, w).cuda())

            with torch.no_grad():
                out = nnModel(varInput)

            outMean = out.view(bs, n_crops, -1).mean(1)

            outPRED = torch.cat((outPRED, outMean.data), 0)

        dataGT = outGT.cpu().numpy()
        dataPRED = outPRED.cpu().numpy()

        AlgorithmClassification.__saveoutput__(dataGT, dataPRED, pathOutputPred)
        AlgorithmClassification.__savescore__(dataGT, dataPRED, pathOutputAcc)
        
    # -------------------------------------------------------------------------------