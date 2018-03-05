#--------------------------------------------------------------------------------
#---- This example shows how to use a DatasetGenerator and ImageTransform classes
#---- DatasetGenerator - class that generates batches from images, defined in a file
#---- ImageTransform - provides useful transforms, that can be fed to DatasetGenerator
#--------------------------------------------------------------------------------

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adadelta
from keras.losses import binary_crossentropy

from ImageTransform import ImageTransform
from DatasetGenerator import DatasetGenerator

#---- Path to a directory where images are located
#---- Internal structure of the directory is not important
pathDatabase = 'D:/ANDREY/Development/deeplearning/database/'

#---- Path to the dataset. For each image this file contains its
#---- path and target vector
pathFile = 'D:/ANDREY/Development/deeplearning/database/dataset1.txt'

imgWidth = 224
imgHeight = 224

batchsize = 16

#---- Transformation sequence. Should end with normalization procedure
transformSequence = []
transformSequence.append(ImageTransform(ImageTransform.TRANSFORM_RESIZE, [imgWidth, imgHeight]))
transformSequence.append(ImageTransform(ImageTransform.TRANSFORM_FLIP_HORIZONTAL))
transformSequence.append(ImageTransform(ImageTransform.TRANSFORM_NORMALIZE))

#---- Define the generator, and give it the transformation sequence
datasetGenerator = DatasetGenerator(pathDatabase, pathFile, transformSequence, imgWidth, imgHeight, batchsize, True)

#---- Create a simple CNN its structure is not important in this example
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(imgWidth, imgHeight, 3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='softmax'))
model.compile(loss=binary_crossentropy, optimizer= Adadelta(), metrics=['accuracy'])

#---- Train the network. Number of epochs, and steps are not important in this example
model.fit_generator(datasetGenerator.generate(), epochs=10, steps_per_epoch=50, shuffle=True, verbose=1)