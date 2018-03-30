import sys, getopt

from TaskManager import TaskManager

#====== SCRIPT ARGUMENTS:
#====== -r  pathRoot   <- this is a path to the database with images
#====== -t  pathTrain  <- this is a path to the dataset file for the training set
#====== -v  pathValid  <- this is a path to the dataset file for the validation set
#====== -e  pathTest   <- this is a path to the dataset file for the testing set
#====== -m  pathOutModel <- this is a directory where the model will be saved
#====== -l  pathOutLog  <- this is a directory where the log file will be saved
#====== -a  pathOutAcc <- this is a directory where the pred/accuracy file will be saved
#====== -n  nnModelType: [alex, inception, vgg, res50, res101, dense121, dense169, dense201]
#====== -i  nnIsTrained: [true, false]
#====== -d nnClassCount <- integer, output dimension
#====== -f   nnLoss: [bce, wbce]
#====== -s   nnActivation [sigmoid, softmax, none]
#====== -q  imgResize (currently only 256)
#====== -p  imgCrop (currently only 224)
#====== -b  trBatch
#====== -x  trMaxEpoch <- integer



def main(argv):

    #---- Define empty containers
    #---- Later we will check if the options will have been set
    pathRoot = None
    pathTrain = None
    pathValid = None
    pathTest = None
    pathOutModel = None
    pathOutLog = None
    pathOutAcc = None
    nnModelType = None
    nnIsTrained = None
    nnClassCount = None
    nnLoss = None
    nnActivation = None
    imgResize = None
    imgCrop = None
    trBatch = None
    trMaxEpoch = None


    opts, args = getopt.getopt(argv, "r:t:v:e:m:l:a:n:i:d:f:s:q:p:b:x:")

    #---- Parse options

    for opt, arg in opts:

        if opt == '-r':   pathRoot = arg
        elif opt == '-t': pathTrain = arg
        elif opt == '-v': pathValid = arg
        elif opt == '-e': pathTest = arg
        elif opt == '-m': pathOutModel = arg
        elif opt == '-l': pathOutLog = arg
        elif opt == '-a': pathOutAcc = arg
        elif opt == '-n':
            if arg == 'alex': nnModelType = TaskManager.MODEL_ALEXNET
            elif arg == 'inception': nnModelType = TaskManager.MODEL_INCEPTION
            elif arg == 'vgg': nnModelType = TaskManager.MODEL_VGGN16
            elif arg == 'res50': nnModelType = TaskManager.MODEL_RESNET50
            elif arg == 'res101': nnModelType = TaskManager.MODEL_RESNET101
            elif arg == 'dense121': nnModelType = TaskManager.MODEL_DENSENET121
            elif arg == 'dense169': nnModelType = TaskManager.MODEL_DENSENET169
            elif arg == 'dense201': nnModelType = TaskManager.MODEL_DENSENET201
        elif opt == '-d': nnClassCount = int(arg)
        elif opt == '-i':
            if arg == 'true': nnIsTrained = True
            else: nnIsTrained = False
        elif opt == '-f':
            if arg == 'bce': nnLoss = TaskManager.LOSS_BCE
            elif arg == 'wbce': nnLoss = TaskManager.LOSS_WBCE
        elif opt == '-s':
            if arg == 'sigmoid': nnActivation = TaskManager.ACTIVATION_SIGMOID
            elif arg == 'softmax': nnActivation = TaskManager.ACTIVATION_SOFTMAX
            elif arg == 'none': nnActivation = TaskManager.ACTIVATION_NONE
        elif opt == '-q': imgResize = int(arg)
        elif opt == '-p': imgCrop = int(arg)
        elif opt == '-b': trBatch = int(arg)
        elif opt == '-x': trMaxEpoch = int(arg)

        checkpoint = None

    #---- Chek if all options are set

    if pathRoot == None:
        print ("Argument -r is not set!")
        exit()
    if pathTrain == None :
        print ("Argument -t is not set!")
        exit()
    if pathValid == None :
        print ("Argument -v is not set!")
        exit()
    if pathTest == None :
        print ("Argument -e is not set!")
        exit()
    if pathOutModel == None :
        print ("Argument -m is not set!")
        exit()
    if pathOutLog == None:
        print ("Argument -l is not set!")
        exit()
    if pathOutAcc == None :
        print ("Argument -a is not set!")
        exit()
    if nnModelType == None :
        print ("Argument -n is not set!")
        exit()
    if nnClassCount == None :
        print ("Argument -d is not set!")
        exit()
    if nnIsTrained == None :
        print ("Argument -i is not set!")
        exit()
    if nnLoss == None:
        print ("Argument -f is not set!")
        exit()
    if nnActivation == None :
        print ("Argument -s is not set!")
        exit()
    if imgResize == None :
        print ("Argument -q is not set!")
        exit()
    if imgCrop == None :
        print ("Argument -p is not set!")
        exit()
    if trBatch == None :
        print ("Argument -b is not set!")
        exit()
    if trMaxEpoch ==None:
        print ('Argument -x is not set!')
        exit()

    #--------------- LAUNCH TASK
    TaskManager.launchTask(pathRoot, pathTrain, pathValid, pathTest, pathOutModel, pathOutLog, pathOutAcc, nnModelType,
                           nnClassCount, nnIsTrained, nnLoss, nnActivation, imgResize, imgCrop, trBatch, trMaxEpoch,
                           checkpoint)

if __name__ == "__main__":
    main(sys.argv[1:])