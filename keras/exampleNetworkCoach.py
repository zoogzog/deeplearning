import time
import gc

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adadelta
from keras.losses import binary_crossentropy


from ImageTransform import ImageTransform
from NetworkCoach import NetworkCoach

#----------------------------------------------------------------------------------------
#---- Create a simple CNN its particular structure is not important in this example
def getModel (imgWidth, imgHeight):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(imgWidth, imgHeight, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='softmax'))
    
    return model

#----------------------------------------------------------------------------------------
#---- This method executes a single task

def executeTaskSingle (task):

    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime

    pathDatabase = '../database/'

    imgWidth = 224
    imgHeight = 224

    #---- Define transformation sequences for training and validation sets
    trSequenceTrain = []
    trSequenceTrain.append(ImageTransform(ImageTransform.TRANSFORM_RESIZE, [imgWidth, imgHeight]))
    trSequenceTrain.append(ImageTransform(ImageTransform.TRANSFORM_FLIP_HORIZONTAL))
    trSequenceTrain.append(ImageTransform(ImageTransform.TRANSFORM_NORMALIZE))
    
    trSequenceVal = []
    trSequenceVal.append(ImageTransform(ImageTransform.TRANSFORM_RESIZE, [imgWidth, imgHeight]))
    trSequenceVal.append(ImageTransform(ImageTransform.TRANSFORM_NORMALIZE))
    
    #---- Load task-specific paramters 
    pathFileTrain = task[0]
    pathFileValidate = task[1]
    pathFileTest = task[2]
    
    nnModel = getModel(imgWidth, imgHeight)
    nnBatchsize = task[3]
    nnEpochs = task[4]
    nnLoss = task[5]
    nnOptimizer = Adadelta()
    nnTargetDim = 1
    
    callbackList = []
    
    pathModel = 'm-' + timestampLaunch + '.h5'
    
    # ======== TRAIN NETWORK ========
    NetworkCoach.netTrainAuto (pathDatabase, pathFileTrain, pathFileValidate, trSequenceTrain, trSequenceVal, imgWidth, imgHeight, nnModel, nnLoss, nnOptimizer, nnBatchsize, nnEpochs, callbackList, pathModel)
    
    # ======== TEST NETWORK ========
    auroc, outputTP = NetworkCoach.netTestAuto (pathDatabase, pathFileTest, trSequenceVal, imgWidth, imgHeight, nnModel, nnBatchsize, nnTargetDim)
    
    return auroc, outputTP, pathModel

#----------------------------------------------------------------------------------------
#---- This method executes all tasks, specified in the list

def executeTaskMultiple (task):
    
    #---- Create log file
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime

    logfile = open('log-exe-' + timestampLaunch + '.txt', 'w')
    
    #========== EXECUTE ALL TASKS ==========
    for i in range (0, len(task)):

        auroc, outputTP, pathModel = executeTaskSingle(task[i])

        #---- Write to log file
        logfile.write('TASK [' + str(i) +']: ')
        for k in range (0, len(task[i])):  logfile.write(str(task[i][k]) + ' ')
        logfile.write('\nMODEL NAME: ' +  pathModel +'\n')
        logfile.write('AUROC: ')

        for k in range (0, len(auroc)): logfile.write(str(auroc[k]) + ' ')
        logfile.write('\nACCURACY: ')
        for k in range (0, len(outputTP)): logfile.write(str(outputTP[k]) + ' ')
        logfile.write('\n')

        logfile.flush()

        gc.collect()

    #=======================================


    logfile.close()

#----------------------------------------------------------------------------------------

LOSS_BINARYCROSSE = 'binary_crossentropy'

#---- Specify parameters of the tasks here, in this array
#---- For this example it is not of importance that training, validataion, tests sets are the same.
task = [
    ['../database/dataset1.txt', '../database/dataset1.txt', '../database/dataset1.txt', 16, 10, LOSS_BINARYCROSSE],
    ['../database/dataset2.txt', '../database/dataset2.txt', '../database/dataset2.txt', 16, 10, LOSS_BINARYCROSSE],
    ]

executeTaskMultiple (task)