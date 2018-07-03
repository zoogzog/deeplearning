import os
import time

import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

#------------ CUSTOM LIBRARIES ------------
import NetworkZoo as Zoo
import LossZoo as Loss

from DatasetGenerator import DatasetGenerator
from NetworkCoach import NetworkCoach



#--------------------------------------------------------------------------------
class TaskManager ():

    #---- Neural Network models that the script can use for training and testing
    NETWORK_NAME = ["ALEX", "INCV3", "VGG16", "RES50", "RES101", "DENSE121", "DENSE169", "DENSE201"]
    MODEL_ALEXNET = 0
    MODEL_INCEPTION = 1
    MODEL_VGGN16 = 2
    MODEL_RESNET50 = 3
    MODEL_RESNET101 = 4
    MODEL_DENSENET121 = 5
    MODEL_DENSENET169 = 6
    MODEL_DENSENET201 = 7

    TRANSFORM_TRAIN = 0
    TRANSFORM_VALID = 1
    TRANSFORM_TEST = 2

    LOSS_NAME = ["BCE", "WBCE"]
    LOSS_BCE = 0
    LOSS_WBCE = 1


    ACTIVATION_SIGMOID = Zoo.ACTIVATION_SIGMOID
    ACTIVATION_SOFTMAX = Zoo.ACTIVATION_SOFTMAX
    ACTIVATION_NONE = Zoo.ACTIVATION_NONE

    # --------------------------------------------------------------------------------
    #---- Build a model, with a specified architecture
    #---- In: modelName - integer, architecture
    #---- In: isTrained - True/False (pre-trained or not on imagenet)
    #---- In: nnActivation - activation of the last layer

    def getModel (modelName, numClasses, isTrained, nnActivation):

        model = None
        if (modelName == TaskManager.MODEL_ALEXNET): model = Zoo.AlexNet(numClasses, isTrained, nnActivation)
        elif (modelName == TaskManager.MODEL_INCEPTION): model = Zoo.Inception(numClasses, isTrained, nnActivation)
        elif (modelName == TaskManager.MODEL_VGGN16): model = Zoo.VGGN16(numClasses, isTrained, nnActivation)
        elif (modelName == TaskManager.MODEL_DENSENET121): model = Zoo.DenseNet121(numClasses, isTrained, nnActivation)
        elif (modelName == TaskManager.MODEL_DENSENET169): model = Zoo.DenseNet169(numClasses, isTrained, nnActivation)
        elif (modelName == TaskManager.MODEL_DENSENET201): model = Zoo.DenseNet201(numClasses, isTrained, nnActivation)
        elif (modelName == TaskManager.MODEL_RESNET50): model = Zoo.ResNet50(numClasses, isTrained, nnActivation)

        return model

    #--------------------------------------------------------------------------------
    #---- Build transformation sequence
    #---- In: sequenceType - type of sequence for training, testing, validation
    #---- In: imgResize -size of the resized image
    #---- In: imgCrop - size of the cropped image

    def getTransformationSequence (sequenceType, imgResize, imgCrop):
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        transformList = []

        if (sequenceType == TaskManager.TRANSFORM_TRAIN):
            transformList.append(transforms.Resize(imgResize))
            transformList.append(transforms.CenterCrop(imgCrop))
            transformList.append(transforms.RandomHorizontalFlip())
            transformList.append(transforms.ToTensor())
            transformList.append(normalize)
            transformSequence = transforms.Compose(transformList)
            return transformSequence

        if (sequenceType == TaskManager.TRANSFORM_VALID):
            transformList.append(transforms.Resize(imgResize))
            transformList.append(transforms.CenterCrop(imgCrop))
            transformList.append(transforms.ToTensor())
            transformList.append(normalize)
            transformSequence = transforms.Compose(transformList)
            return transformSequence

        if (sequenceType == TaskManager.TRANSFORM_TEST):
            transformList = []
            transformList.append(transforms.Resize(imgResize))
            transformList.append(transforms.TenCrop(imgCrop))
            transformList.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
            transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
            return transforms.Compose(transformList)

        return None

    # --------------------------------------------------------------------------------
    #---- Build a specific loss, specified by its name

    def getLoss (lossType, weights=None):
        loss = None

        if (lossType == TaskManager.LOSS_BCE): return torch.nn.BCELoss(size_average=True)
        if (lossType == TaskManager.LOSS_WBCE): return Loss.WeightedBinaryCrossEntropy(weights[0], 1 - weights[0])

    #--------------------------------------------------------------------------------
    #---- Procedure for training during each epoch

    def epochTrain(model, dataLoader, optimizer, scheduler, epochMax, classCount, loss):

        model.train()

        for batchID, (input, target) in enumerate(dataLoader):
            target = target.cuda(async=True)

            varInput = torch.autograd.Variable(input)
            varTarget = torch.autograd.Variable(target)
            varOutput = model(varInput)

            lossvalue = loss(varOutput, varTarget)

            optimizer.zero_grad()
            lossvalue.backward()
            optimizer.step()

    # --------------------------------------------------------------------------------
    #----- Procedure for validation during each epoch

    def epochVal(model, dataLoader, optimizer, scheduler, epochMax, classCount, loss):

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


    #--------------------------------------------------------------------------------
    #---- This method launches a trainin & testing task, collects results, saves in a log file
    #---- In: pathRoot - path to the root directory with the image database
    #---- In: pathTrain  - path to the file, that describes training dataset
    #---- In: pathVal - path to the file, that describes validation dataset
    #---- In: pathTest - path to the file, that describes testing dataset
    #---- In: nnModelType - network architecture to use
    #---- In: nnNumClasses - number of classes (output dimension)
    #---- In: nnIsTrained - is use a network trained on the imagenet
    #---- In: nnLoss - type of loss to use
    #---- In: nnActivation - type of the last activation layer
    #---- In: imgResize - size of the resized image for transformations
    #---- In: imgCrop - size of the resized image for transformations
    #---- In: trBatch - size of the batch

    def launchTask (pathRoot, pathTrain, pathVal, pathTest, pathOutModel, pathOutLog, pathOutAcc, nnModelType, nnNumClasses, nnIsTrained, nnLoss, nnActivation, imgResize, imgCrop, trBatch, nnEpochs, checkpoint):

        #----- Set the output
        timestampTime = time.strftime("%H%M%S")
        timestampDate = time.strftime("%d%m%Y")
        timestampStart = timestampDate + '-' + timestampTime

        fileModel = "m-" + timestampStart + ".pth.tar"
        fileLog = "log-" + timestampStart + ".txt"
        fileAcc = "acc-" + timestampStart + ".txt"
        filePred = "pred-" + timestampStart + ".txt"

        pathOutputModel = os.path.join(pathOutModel, fileModel)
        pathOutputLog = os.path.join(pathOutLog, fileLog)
        pathOutputAcc = os.path.join(pathOutAcc, fileAcc)
        pathOutputPred = os.path.join(pathOutAcc, filePred)

        TaskManager.saveInfo (pathOutputLog, pathRoot, pathTrain, pathVal, pathTest, pathOutModel, pathOutLog, pathOutAcc, nnModelType, nnNumClasses, nnIsTrained, nnLoss, nnActivation, imgResize, imgCrop, trBatch, nnEpochs, checkpoint)

        # -------------------- SETTINGS: NETWORK MODEL
        model = TaskManager.getModel(nnModelType, nnNumClasses, nnIsTrained, nnActivation)
        model = torch.nn.DataParallel(model).cuda()

        # -------------------- SETTINGS: TRANSFORMATIONS
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        transformSequenceTrain = TaskManager.getTransformationSequence(TaskManager.TRANSFORM_TRAIN, imgResize, imgCrop)
        transformSequenceValid = TaskManager.getTransformationSequence(TaskManager.TRANSFORM_VALID, imgResize, imgCrop)
        transformSequenceTest = TaskManager.getTransformationSequence(TaskManager.TRANSFORM_TEST, imgResize, imgCrop)

        # -------------------- SETTINGS: DATASET BUILDERS
        datasetTrain = DatasetGenerator(pathImageDirectory=pathRoot, pathDatasetFile=pathTrain, transform=transformSequenceTrain)
        datasetVal = DatasetGenerator(pathImageDirectory=pathRoot, pathDatasetFile=pathVal, transform=transformSequenceValid)
        datasetTest = DatasetGenerator(pathImageDirectory=pathRoot, pathDatasetFile=pathTest, transform=transformSequenceTest)

        #----- FIXME: On windows machines the num_workers parameter should be equal 0! Or memory leakage/does not launch
        #----- FIXME: On linux machines num_workers can be higher than 8 and even as high as 24, depending on the CPU threading capabilities
        dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatch, shuffle=True, num_workers=0, pin_memory=False)
        dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=trBatch, shuffle=False, num_workers=0, pin_memory=False)
        dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=trBatch, shuffle=False, num_workers=0, pin_memory=False)


        #----- This is the ratio of positive (non zero) samples to all samples in the dataset
        classDistributionTrain = datasetTrain.getClassDistribution()

        # -------------------- SETTINGS: OPTIMIZER & SCHEDULER
        optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5, mode='min')

        # -------------------- SETTINGS: LOSS
        loss = TaskManager.getLoss(nnLoss, weights=classDistributionTrain)

        # ---- Load checkpoint
        if checkpoint != None:
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])
            optimizer.load_state_dict(modelCheckpoint['optimizer'])

        #----- Launch training
        NetworkCoach.netTrain(dataLoaderTrain, dataLoaderVal, model, nnEpochs, loss, optimizer, scheduler, pathOutputModel, pathOutputLog)

        #---- Get the best model
        #modelCheckpoint = torch.load(pathOutputModel)
        #model.load_state_dict(modelCheckpoint['state_dict'])

        #---- Launch testing
        #NetworkCoach.netTest(dataLoaderTest, model, nnNumClasses, pathOutputAcc, pathOutputPred)

    # --------------------------------------------------------------------------------

    def saveInfo (pathOutputLog, pathRoot, pathTrain, pathVal, pathTest, pathOutModel, pathOutLog, pathOutAcc, nnModelType, nnNumClasses, nnIsTrained, nnLoss, nnActivation, imgResize, imgCrop, trBatch, nnEpochs, checkpoint):

        fileLog = open(pathOutputLog, 'w')
        fileLog.write("---------------------------------------------------------\n")
        fileLog.write("DATABASE: " + pathRoot + "\n")
        fileLog.write("DATASET TRAIN: " + pathTrain + "\n")
        fileLog.write("DATASET VALIDATION: " + pathVal + "\n")
        fileLog.write("DATASET TEST: " + pathTest + "\n")
        fileLog.write("OUTPUT MODEL: " + pathOutModel + "\n")
        fileLog.write("OUTPUT LOG: " + pathOutLog + "\n")
        fileLog.write("OUTPUT ACC: " + pathOutAcc + "\n")
        fileLog.write("NETWORK: " + TaskManager.NETWORK_NAME[nnModelType] + "\n")
        fileLog.write("DIMENSION: " + str(nnNumClasses) + "\n")
        fileLog.write("IS TRAINED: " + str(nnIsTrained) + "\n")
        fileLog.write("LOSS: " + TaskManager.LOSS_NAME[nnLoss] + "\n")
        fileLog.write("ACTIVATION ID: " + str(nnActivation) + "\n")
        fileLog.write("TRANSFORM RES: " + str(imgResize) + "\n")
        fileLog.write("TRANSFORM CROP: " + str(imgCrop) + "\n")
        fileLog.write("BATCH: " + str(trBatch) + "\n")
        fileLog.write("EPOCHS: " + str(nnEpochs) + "\n")
        fileLog.write("---------------------------------------------------------\n")
        fileLog.flush()
        fileLog.close()
#--------------------------------------------------------------------------------


