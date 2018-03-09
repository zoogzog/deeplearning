import keras
import numpy as np
import os
import time

#-------------------------------

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler
from keras.models import load_model

from EnvironmentSettings import EnvironmentSettings

#--------------------------------------------------------------------------------
#---- This class is a callback for the NN training process with Keras
#---- This callback saves accuracy and loss after each iteration in a log file
#---- The name of the log file is the logger-$date-$time.txt
#--------------------------------------------------------------------------------

class CallbackLogger(Callback):
    
    #---- Internall variables:
    #---- logger - descriptor of the log file
    #---- acc - array of accuracies
    #---- losses - array of losses
    
    def __init__ (self, modelName):
        self.logName = modelName
    
    
    #--------------------------------------------------------------------------------
    #---- This is executed when training is started
    
    def on_train_begin(self, logs={}):
        
        #---- Generate timestamp
        timestampTime = time.strftime("%H%M%S")
        timestampDate = time.strftime("%d%m%Y")

        #---- Save time and date when training has been started
        loggerFilePath = 'log-' + self.logName + '.txt'
        self.logger = open(loggerFilePath, 'a')
        self.logger.write('Training started at ' + timestampDate + '-' + timestampTime + '\n')
        self.logger.flush()
        self.epochcounter = 0
        
        self.acctrain = []
        self.losstrain = []
        
        self.accval = []
        self.lossval = []

    #--------------------------------------------------------------------------------
    #---- This is executed when training is finished

    def on_train_end(self, logs={}):
        timestampTime = time.strftime("%H%M%S")
        timestampDate = time.strftime("%d%m%Y")
        
        self.logger.write('Training finished at ' + timestampDate + '-' + timestampTime + '\n')
        self.logger.close()
        
        return
    
    #--------------------------------------------------------------------------------
    #---- This is executed when new epoch is started

    def on_epoch_begin(self, epoch, logs={}):
        return

    #--------------------------------------------------------------------------------
    #---- This is executed when an epoch ends

    def on_epoch_end(self, epoch, logs={}):

        #---- Generate timestamp
        timestampTime = time.strftime("%H%M%S")
        timestampDate = time.strftime("%d%m%Y")

        #---- Get loss and accuracy        
        trainLoss = logs.get('loss')
        trainAccuracy = logs.get('acc')
        
        valLoss = logs.get('val_loss')
        valAccuracy = logs.get('val_acc')
        
        #---- Get the index of the current epoch, write to the log file
        self.epochcounter = self.epochcounter + 1
        self.logger.write(timestampDate + '-' + timestampTime + ' EPOCH '  + str(self.epochcounter) + ': t_acc=' + str(trainAccuracy) + ' t_loss=' + str(trainLoss) + 
                                                                                                    ' v_acc=' + str(valAccuracy) + ' v_loss=' + str(valLoss) + '\n')
        self.logger.flush()
        
        #---- Append the arrays
        self.acctrain.append(trainLoss)
        self.losstrain.append(trainAccuracy)
        self.accval.append(valAccuracy)
        self.lossval.append(valLoss)
        return

    #--------------------------------------------------------------------------------
    #---- This is executed when a new batch is started

    def on_batch_begin(self, batch, logs={}):
        return

    #--------------------------------------------------------------------------------
    #---- This is executed when a batch training has ended

    def on_batch_end(self, batch, logs={}):
        return
    
#--------------------------------------------------------------------------------