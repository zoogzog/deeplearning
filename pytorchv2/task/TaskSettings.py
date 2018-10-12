import torch
import torchvision.transforms as transforms
import sys

sys.path.append('../')
from network.NetZoo import *
from network.NetLoss import *
from network.NetDenseBig import *
from datagen.DatagenClassification import  *
from datagen.DatagenOversampling import *
from datagen.DatagenAutoencoder import *

class TaskSettings():

    # ==================== CONSTANTS ====================

    # -------------------- NETWORKS ---------------------
    NN_DENSENET121 = 0
    NN_DENSENET169 = 1
    NN_DENSENET201 = 2
    NN_ALEXNET = 3
    NN_RESNET50 = 4
    NN_RESNET101 = 5
    NN_INCEPTION = 6
    NN_VGGN16 = 7
    NN_DENSENET121BIG = 8

    @staticmethod
    def NN_GETNAME(index):
        namelist=["DENSENET121", "DENSENET169", "DENSENET201", "ALEXNET", "RESNET50",
                  "RESNET101", "INCEPTION", "VGGN16","DENSENET121BIG"]
        if index >= 0 and index < len(namelist):
            return namelist[index]
        return ""

    def NN_GETID(name):
        namelist = ["DENSENET121", "DENSENET169", "DENSENET201", "ALEXNET", "RESNET50",
                    "RESNET101", "INCEPTION", "VGGN16","DENSENET121BIG"]
        nameupper = name.upper()

        return namelist.index(nameupper)


    # -------------------- ACTIVATION --------------------
    ACTIVATION_SIGMOID = 0
    ACTIVATION_SOFTMAX = 1
    ACTIVATION_NONE = 2

    def ACTIVATION_GETNAME(index):
        namelist=["SIGMOID", "SOFTMAX", "NONE"]
        if index >= 0 and index < len(namelist):
            return namelist[index]
        return ""

    def ACTIVATION_GETID(name):
        namelist = ["SIGMOID", "SOFTMAX", "NONE"]
        nameupper = name.upper()
        return namelist.index(nameupper)

    # ------------------ TRANSFORMATION ------------------

    TRANSFORM_RANDCROP = 0
    TRANSFORM_RESIZE = 1
    TRANSFORM_CCROP = 2
    TRANSFORM_10CROP = 3

    def TRANSFORM_GETNAME(index):
        namelist=["RNDCROP", "RESIZE", "CCROP", "10CROP"]
        if index >= 0 and index < len(namelist):
            return namelist[index]
        return ""

    def TRANSFORM_GETID(name):
        namelist = ["RNDCROP", "RESIZE", "CCROP", "10CROP"]
        nameupper = name.upper()
        return namelist.index(nameupper)

    # ------------------------ LOSS ----------------------

    LOSS_BCE = 0
    LOSS_WBCE = 1
    LOSS_WBCEMC = 2

    def LOSS_GETNAME(index):
        namelist=["BCE", "WBCE", "WBCEMC"]
        if index >= 0 and index < len(namelist):
            return namelist[index]
        return ""

    def LOSS_GETID(name):
        namelist = ["BCE", "WBCE", "WBCEMC"]
        nameupper = name.upper()
        return namelist.index(nameupper)

    # ---------------------- TASK TYPE -------------------

    TASK_TYPE_CLASSIFICATION = 0
    TASK_TYPE_SEGMENTATION = 1

    DATAGEN_BASE = 0
    DATAGEN_OVERSAMPLE = 1
    DATAGEN_AUTOENCODER = 2

    def DATAGEN_GETNAME(index):
        namelist = ["CLASSBASE", "CLASSOVERSAMPLE", "AUTOENCODER"]
        if index >= 0 and index < len(namelist):
            return namelist[index]
        return ""

    def DATAGEN_GETID(name):
        namelist = ["CLASSBASE", "CLASSOVERSAMPLE", "AUTOENCODER"]
        nameupper = name.upper()
        return namelist.index(nameupper)

    # ------------------------------------ PRIVATE -----------------------------------
    def __init__(self):
        self.NETWORK_TYPE = 0
        self.NETWORK_ISTRAINED = False
        self.NETWORK_CLASSCOUNT = 0

        self.ACTIVATION_TYPE = 0
        self.LEARNING_RATE = 0.001

        self.TRANSFORM_SEQUENCE_TRAIN = []
        self.TRANSFORM_SEQUENCE_TRAIN_PARAMETERS = []

        self.TRANSFORM_SEQUENCE_VALIDATE = []
        self.TRANSFORM_SEQUENCE_VALIDATE_PARAMETERS = []

        self.TRANSFORM_SEQUENCE_TEST = []
        self.TRANSFORM_SEQUENCE_TEST_PARAMETERS = []

        self.LOSS = 0
        self.EPOCH = 0
        self.BATCH = 0

        self.PATH_IN_ROOT = ""
        self.PATH_IN_TRAIN = ""
        self.PATH_IN_VALIDATE = ""
        self.PATH_IN_TEST = ""

        self.PATH_OUT_LOG = ""
        self.PATH_OUT_MODEL = ""
        self.PATH_OUT_ACCURACY = ""

        self.TASK_TYPE = 0
        self.DATAGEN_TYPE = 0
        self.DATAGEN_TRAIN = None
        self.DATAGEN_VALIDATION = None
        self.DATAGEN_TEST = None

    # ------------------------------------ PUBLIC ------------------------------------

    def getsettings(self):
        """
        Get all the specified settings
        :return: (string) - a string with current settings
        """

        transformTrain = ""
        for i in range(0, len(self.TRANSFORM_SEQUENCE_TRAIN)):
            transformTrain += "[" + TaskSettings.TRANSFORM_GETNAME(self.TRANSFORM_SEQUENCE_TRAIN[i]) + "," + \
                                    str(self.TRANSFORM_SEQUENCE_TRAIN_PARAMETERS[i]) + "] "

        transformValidation = ""
        for i in range(0, len(self.TRANSFORM_SEQUENCE_VALIDATE)):
            transformValidation += "[" + TaskSettings.TRANSFORM_GETNAME(self.TRANSFORM_SEQUENCE_VALIDATE[i]) + "," + \
                                    str(self.TRANSFORM_SEQUENCE_VALIDATE_PARAMETERS[i]) + "] "

        transformTest = ""
        for i in range(0, len(self.TRANSFORM_SEQUENCE_TEST)):
            transformTest += "[" + TaskSettings.TRANSFORM_GETNAME(self.TRANSFORM_SEQUENCE_TEST[i]) + "," + \
                                    str(self.TRANSFORM_SEQUENCE_TEST_PARAMETERS[i]) + "] "

        outputstr = ""
        outputstr += "DATABASE: " + self.PATH_IN_ROOT + "\n"
        outputstr += "DATASET TRAIN: " + self.PATH_IN_TRAIN + "\n"
        outputstr += "DATASET VALIDATE: " + self.PATH_IN_VALIDATE + "\n"
        outputstr += "DATASET TEST: " + self.PATH_IN_TEST + "\n"
        outputstr += "DATAGENERATOR: " + TaskSettings.DATAGEN_GETNAME(self.DATAGEN_TYPE) + "\n"
        outputstr += "TASK TYPE: " + str(self.TASK_TYPE) + "\n"
        outputstr += "OUTPUT LOG: " + self.PATH_OUT_LOG + "\n"
        outputstr += "OUTPUT MODEL: " + self.PATH_OUT_MODEL + "\n"
        outputstr += "OUTPUT ACCURACY: " + self.PATH_OUT_ACCURACY + "\n"
        outputstr += "NETWORK: " + TaskSettings.NN_GETNAME(self.NETWORK_TYPE) + "\n"
        outputstr += "NETWORK CLASS COUNT: " + str(self.NETWORK_CLASSCOUNT) + "\n"
        outputstr += "NETOWRK PRE-TRAINED: " + str(self.NETWORK_ISTRAINED) + "\n"
        outputstr += "ACTIVATION: " + TaskSettings.ACTIVATION_GETNAME(self.ACTIVATION_TYPE) + "\n"
        outputstr += "LOSS: " + TaskSettings.LOSS_GETNAME(self.LOSS) + "\n"
        outputstr += "LEARNING RATE: " + str(self.LEARNING_RATE) + "\n"
        outputstr += "TRANSFORM SEQUENCE [TRAIN]: " + transformTrain + "\n"
        outputstr += "TRANSFORM SEQUENCE [VALID]: " + transformValidation + "\n"
        outputstr += "TRANSFORM SEQUENCE [TEST]: " + transformTest + "\n"
        outputstr += "TRAINING EPOCHS: " + str(self.EPOCH) + "\n"
        outputstr += "BATCH SIZE: " + str(self.BATCH)

        return outputstr

    # --------------------------------------------------------------------------------

    def getNetworkType(self):
        """
        Get the type of the network
        :return: (int) - id of the network. use NN_GETNAME(int) to get it's name
        """
        return self.NETWORK_TYPE

    def getNetwork(self):
        if self.NETWORK_TYPE == TaskSettings.NN_DENSENET121: return DenseNet121(self.NETWORK_CLASSCOUNT, self.NETWORK_ISTRAINED, self.getActivation())
        if self.NETWORK_TYPE == TaskSettings.NN_DENSENET169: return DenseNet169(self.NETWORK_CLASSCOUNT, self.NETWORK_ISTRAINED, self.getActivation())
        if self.NETWORK_TYPE == TaskSettings.NN_DENSENET201: return DenseNet201(self.NETWORK_CLASSCOUNT, self.NETWORK_ISTRAINED, self.getActivation())
        if self.NETWORK_TYPE == TaskSettings.NN_ALEXNET: return AlexNet(self.NETWORK_CLASSCOUNT, self.NETWORK_ISTRAINED, self.getActivation())
        if self.NETWORK_TYPE == TaskSettings.NN_RESNET50: return ResNet50(self.NETWORK_CLASSCOUNT, self.NETWORK_ISTRAINED, self.getActivation())
        if self.NETWORK_TYPE == TaskSettings.NN_RESNET101: return ResNet101(self.NETWORK_CLASSCOUNT, self.NETWORK_ISTRAINED, self.getActivation())
        if self.NETWORK_TYPE == TaskSettings.NN_INCEPTION: return Inception(self.NETWORK_CLASSCOUNT, self.NETWORK_ISTRAINED, self.getActivation())
        if self.NETWORK_TYPE == TaskSettings.NN_VGGN16: return VGGN16(self.NETWORK_CLASSCOUNT, self.NETWORK_ISTRAINED, self.getActivation())
        if self.NETWORK_TYPE == TaskSettings.NN_DENSENET121BIG: return BigDensenet121(self.NETWORK_CLASSCOUNT, self.NETWORK_ISTRAINED, self.getActivation())


    def getAcitvationType(self):
        """
        Get the type of the activation function
        :return: (int) - id of the activation function
        """
        return self.ACTIVATION_TYPE

    def getActivation(self):
        if self.ACTIVATION_TYPE == TaskSettings.ACTIVATION_SIGMOID: return ACTIVATION_SIGMOID
        if self.ACTIVATION_TYPE == TaskSettings.ACTIVATION_SOFTMAX: return ACTIVATION_SOFTMAX
        if self.ACTIVATION_TYPE == TaskSettings.ACTIVATION_NONE: return ACTIVATION_NONE

        return ACTIVATION_NONE

    def getLearningRate(self):
        """
        Get learning rate
        :return: (float) - learning rate
        """
        return self.LEARNING_RATE

    def getTransformTrain(self):
        """
        Get the transformation sequence for the training set
        :return: (transformation sequence) structure to be used by the loader
        """
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []

        for i in range(0, len(self.TRANSFORM_SEQUENCE_TRAIN)):
            if self.TRANSFORM_SEQUENCE_TRAIN[i] == TaskSettings.TRANSFORM_RANDCROP:
                transformList.append(transforms.RandomResizedCrop(self.TRANSFORM_SEQUENCE_TRAIN_PARAMETERS[i]))
            if self.TRANSFORM_SEQUENCE_TRAIN[i] == TaskSettings.TRANSFORM_RESIZE:
                transformList.append(transforms.Resize(self.TRANSFORM_SEQUENCE_TRAIN_PARAMETERS[i]))
            if self.TRANSFORM_SEQUENCE_TRAIN[i] == TaskSettings.TRANSFORM_CCROP:
                transformList.append(transforms.CenterCrop(self.TRANSFORM_SEQUENCE_TRAIN_PARAMETERS[i]))

        transformList.append(transforms.ToTensor())
        transformList.append(normalize)

        transformSequence = transforms.Compose(transformList)
        return transformSequence

    def getTransformValidation(self):
        """
        Get the transformation sequence for the validataion set
        :return: (transformation sequence) structure to be used by the loader
        """
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []

        for i in range(0, len(self.TRANSFORM_SEQUENCE_TRAIN)):
            if self.TRANSFORM_SEQUENCE_VALIDATE[i] == TaskSettings.TRANSFORM_RANDCROP:
                transformList.append(transforms.RandomResizedCrop(self.TRANSFORM_SEQUENCE_VALIDATE_PARAMETERS[i]))
            if self.TRANSFORM_SEQUENCE_VALIDATE[i] == TaskSettings.TRANSFORM_RESIZE:
                transformList.append(transforms.Resize(self.TRANSFORM_SEQUENCE_VALIDATE_PARAMETERS[i]))
            if self.TRANSFORM_SEQUENCE_VALIDATE[i] == TaskSettings.TRANSFORM_CCROP:
                transformList.append(transforms.CenterCrop(self.TRANSFORM_SEQUENCE_VALIDATE_PARAMETERS[i]))

        transformList.append(transforms.ToTensor())
        transformList.append(normalize)

        transformSequence = transforms.Compose(transformList)
        return transformSequence

    def getTransformTest(self):
        """
         Get the transformation sequence for the test set
         :return: (transformation sequence) structure to be used by the loader
         """
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []

        istencrop = False
        for i in range(0, len(self.TRANSFORM_SEQUENCE_TRAIN)):
            if self.TRANSFORM_SEQUENCE_TEST[i] == TaskSettings.TRANSFORM_RANDCROP:
                transformList.append(transforms.RandomResizedCrop(self.TRANSFORM_SEQUENCE_TEST_PARAMETERS[i]))
            if self.TRANSFORM_SEQUENCE_TEST[i] == TaskSettings.TRANSFORM_RESIZE:
                transformList.append(transforms.Resize(self.TRANSFORM_SEQUENCE_TEST_PARAMETERS[i]))
            if self.TRANSFORM_SEQUENCE_TEST[i] == TaskSettings.TRANSFORM_CCROP:
                transformList.append(transforms.CenterCrop(self.TRANSFORM_SEQUENCE_TEST_PARAMETERS[i]))
            if self.TRANSFORM_SEQUENCE_TEST[i] == TaskSettings.TRANSFORM_10CROP:
                transformList.append(transforms.TenCrop(self.TRANSFORM_SEQUENCE_TEST_PARAMETERS[i]))
                istencrop = True

        if istencrop:
            transformList.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
            transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        else:
            transformList.append(transforms.ToTensor())
            transformList.append(normalize)

        transformSequence = transforms.Compose(transformList)
        return transformSequence

    def getLossType(self):
        return self.LOSS

    def getLoss(self, weights = None):

        # ---- Binary cross entropy loss for binary classification
        if self.LOSS == TaskSettings.LOSS_BCE:
            return torch.nn.BCELoss(size_average=True)

        # ---- Weighted binary cross entropy for binary classification with unbalanced datasets
        if self.LOSS == TaskSettings.LOSS_WBCE:
            if weightlist != None:
                return WeightedBinaryCrossEntropy(weights[0], 1 - weights[0])
            else:
                return None

        # ---- Weighted binary cross entropy for multi-class classification
        if self.LOSS == TaskSettings.LOSS_WBCEMC:
            if weightlist != None:
                return WeightedBinaryCrossEntropyMC(weights)
            else:
                return None

    def getPathInRoot(self):
        return self.PATH_IN_ROOT

    def getPathInTrain(self):
        return self.PATH_IN_TRAIN

    def getPathInValidate(self):
        return self.PATH_IN_VALIDATE

    def getPathInTest(self):
        return self.PATH_IN_TEST

    def getPathOutLog(self):
        return self.PATH_OUT_LOG

    def getPathOutModel(self):
        return self.PATH_OUT_MODEL

    def getPathOutAccuracy(self):
        return self.PATH_OUT_ACCURACY

    def getEpoch(self):
        return self.EPOCH

    def getTaskType(self):
        return self.TASK_TYPE

    def getDatagenType(self):
        return self.DATAGEN_TYPE

    def getBatchSize(self):
        return self.BATCH

    def getDatageneratorTrain(self):
        if self.DATAGEN_TYPE == 0: return DatagenClassification(self.PATH_IN_ROOT, self.PATH_IN_TRAIN, self.getTransformTrain())
        if self.DATAGEN_TYPE == 1: return DatagenOversampling(self.PATH_IN_ROOT, self.PATH_IN_TRAIN, self.getTransformTrain())
        if self.DATAGEN_TYPE == 2: return DatagenAutoencoder(self.PATH_IN_ROOT, self.PATH_IN_TRAIN, self.getTransformTrain(), self.getTransformTrain())

    def getDatageneratorValidation(self):
        if self.DATAGEN_TYPE == 0: return DatagenClassification(self.PATH_IN_ROOT, self.PATH_IN_VALIDATE, self.getTransformValidation())
        if self.DATAGEN_TYPE == 1: return DatagenOversampling(self.PATH_IN_ROOT, self.PATH_IN_VALIDATE, self.getTransformValidation())
        if self.DATAGEN_TYPE == 2: return DatagenAutoencoder(self.PATH_IN_ROOT, self.PATH_IN_VALIDATE, self.getTransformValidation(), self.getTransformValidation())

    def getDatageneratorTest(self):
        if self.DATAGEN_TYPE == 0: return DatagenClassification(self.PATH_IN_ROOT, self.PATH_IN_TEST, self.getTransformTest())
        if self.DATAGEN_TYPE == 1: return DatagenOversampling(self.PATH_IN_ROOT, self.PATH_IN_TEST, self.getTransformTest())
        if self.DATAGEN_TYPE == 2: return DatagenAutoencoder(self.PATH_IN_ROOT, self.PATH_IN_TEST, self.getTransformTest(), self.getTransformTest())


    # --------------------------------------------------------------------------------

    def setNetworkType(self, networkID):
        """
        Set an id of the network. No checks done
        :param networkID: (int) network id from the list of networks
        """
        self.NETWORK_TYPE = networkID

    def setNetworkIsTrained(self, istrained):
        """
        Specify if the network is pre-trained or not
        :param istrained: True or False
        """
        self.NETWORK_ISTRAINED = istrained

    def setNetworkClassCount(self, classcount):
        """
        Set the number of classes
        :param classcount: number of classes
        """
        self.NETWORK_CLASSCOUNT = classcount

    def setActivationType(self, activationID):
        """
        Set an id of the activation function for the network
        :param activationID: (int) activation function id
        """
        self.ACTIVATION_TYPE = activationID

    def setLearningRate(self, learningrate):
        """
        Set the learning rate for the training procedure
        :param learningrate: (double)
        """
        self.LEARNING_RATE = learningrate

    def setTransformSequenceTrain(self, sequence, isname = False):
        """
        :param sequence: sequence of transformations composed of IDs or names
        :param isname: if True then the given sequence contains names
        """
        if isname == True:
            temp = []

            for i in range(0, len(sequence)):
                temp.append(TaskSettings.TRANSFORM_GETID(sequence[i]))

            self.TRANSFORM_SEQUENCE_TRAIN = temp
        else:
            self.TRANSFORM_SEQUENCE_TRAIN = sequence

    def setTransformSequenceTrainParameters(self, sequence):
        self.TRANSFORM_SEQUENCE_TRAIN_PARAMETERS = sequence

    def setTransformSequenceValidate(self, sequence, isname =False):
        """
        :param sequence: sequence of transformations composed of IDs or names
        :param isname: if True then the given sequence contains names
        """
        if isname == True:
            temp = []

            for i in range(0, len(sequence)):
                temp.append(TaskSettings.TRANSFORM_GETID(sequence[i]))

            self.TRANSFORM_SEQUENCE_VALIDATE = temp
        else:
            self.TRANSFORM_SEQUENCE_VALIDATE = sequence

    def setTransformSequenceValidateParameters(self, sequence):
        self.TRANSFORM_SEQUENCE_VALIDATE_PARAMETERS = sequence

    def setTransformSequenceTest(self, sequence, isname = False):
        """
                :param sequence: sequence of transformations composed of IDs or names
                :param isname: if True then the given sequence contains names
                """
        if isname == True:
            temp = []

            for i in range(0, len(sequence)):
                temp.append(TaskSettings.TRANSFORM_GETID(sequence[i]))

            self.TRANSFORM_SEQUENCE_TEST = temp
        else:
            self.TRANSFORM_SEQUENCE_TEST = sequence

    def setTransformSequenceTestParameters(self, sequence):
        self.TRANSFORM_SEQUENCE_TEST_PARAMETERS = sequence


    def setLoss(self, lossID):
        """
        Set the loss function via its ID
        :param lossID: the id of the loss function
        """
        self.LOSS

    def setEnvironmentPaths(self, pathInRoot, pathInTrain, pathInValidate, pathInTest, pathOutLog, pathOutModel, pathOutAccuracy):
        """
        Set all paths to input and output directories and files
        :param pathInRoot: path to the database folder, where images are located
        :param pathInTrain: path to the training dataset file
        :param pathInValidate: path to the validation dataset file
        :param pathInTest: path to the test dataset file
        :param pathOutLog: path to the directory where the log file will be saved
        :param pathOutModel: path to the directory where the model file will be saved
        :param pathOutAccuracy: path to the directory where the accuracy file will be saved
        """
        self.PATH_IN_ROOT = pathInRoot
        self.PATH_IN_TRAIN = pathInTrain
        self.PATH_IN_VALIDATE = pathInValidate
        self.PATH_IN_TEST = pathInTest

        self.PATH_OUT_LOG = pathOutLog
        self.PATH_OUT_MODEL = pathOutModel
        self.PATH_OUT_ACCURACY = pathOutAccuracy

    def setPathInRoot(self, path):
        self.PATH_IN_ROOT = path

    def setPathInTrain(self, path):
        self.PATH_IN_TRAIN = path

    def setPathInValidate(self, path):
        self.PATH_IN_VALIDATE = path

    def setPathInTest(self, path):
        self.PATH_IN_TEST = path

    def setPathOutLog(self, path):
        self.PATH_OUT_LOG = path

    def setPathOutModel(self, path):
        self.PATH_OUT_MODEL = path

    def setPathOutAccuracy(self, path):
        self.PATH_OUT_ACCURACY = path

    def setEpoch(self, value):
        self.EPOCH = value

    def setTaskType(self, value):
        self.TASK_TYPE = value

    def setDatagenType(self, value):
        self.DATAGEN_TYPE = value

    def setBatchSize(self, value):
        self.BATCH = value

    # --------------------------------------------------------------------------------

    def loadDataMap (self, table):

        setNetworkType(table["netid"])
        setNetworkIsTrained(table["netistrained"])
        setNetworkClassCount(table["netclasscount"])
