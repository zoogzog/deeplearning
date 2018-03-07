import os
import keras
import numpy as np
#--------------------------------------------------------------------------------

from keras.preprocessing.image import ImageDataGenerator
from PIL import Image

import ImageTransform

#--------------------------------------------------------------------------------

#---- This class parsers a dataset file, which has the following format
#---- <$image-file-path $target-vector>
#---- * $image-file-path: is the path to an image in a database
#---- This path can be absolute or relative. If relative path is used, then
#---- the root directory path to image database has to be specified in the constructor.
#---- * $target-vector: is a vector of integers, separated with spaces
#---- No checks done to insure the correct data format of the input file

class DatasetGenerator (ImageDataGenerator):

    #--------------------------------------------------------------------------------
    #---- Initialize the dataset generator
    #---- pathImageDatabase - root directory of the image database
    #---- pathDatasetFile - path to the dataset file
    #---- imgWidth - width of the transformed image
    #---- imgHeight - height of the transformed image
    #---- batchsize - size of the batch to generate
    #---- isShuffle - boolean, do data shuffling or not

    def __init__(self, pathImageDatabase, pathDatasetFile, transformList, imgWidth, imgHeight, batchsize, isShuffle):

        self.transformList = transformList
        self.listImagePaths = []
        self.listImageLabels = []
        self.imgWidth = imgWidth
        self.imgHeight = imgHeight
        self.batchsize = batchsize
        self.isShuffle = isShuffle

        inStream = open(pathDatasetFile, "r")

        #---- Scan the file and extract data
        line = True

        while line:

            line = inStream.readline()

            #---- If the line is not empty, then parse it
            if line:
                lineItems = line.split()

                #---- Here we are lazy and do not check if the
                #---- obtained data is correctly formated
                imagePath = os.path.join(pathImageDatabase, lineItems[0])
                imageLabel = lineItems[1:]
                imageLabel = [int(i) for i in imageLabel]

                self.listImagePaths.append(imagePath)
                self.listImageLabels.append(imageLabel)

    #--------------------------------------------------------------------------------
    #---- Return: size of the dataset

    def getSize (self):
        return (len(self.listImagePaths))

    #--------------------------------------------------------------------------------
    #---- Generates indexes in range [0, samples max count]
    #---- Order of these indexes is randomized

    def generateIndexList (self):

        indexList = np.arange(len(self.listImagePaths))

        if self.isShuffle == True:
            np.random.shuffle(indexList)

        return indexList

    #--------------------------------------------------------------------------------
    #---- Generates a batch from images
    #--- indexList - indexes of image paths from the path list

    def generateBatch (self, indexList):

        #---- Define two containers for storing batches of images and target vectors
        #---- Lazy intialization for Y's second dimension here
        X = np.zeros((self.batchsize, self.imgWidth, self.imgHeight, 3))
        Y = np.zeros((self.batchsize, len(self.listImageLabels[0])))

        #---- Generate batch of the specified length
        for batchID in range (0, self.batchsize):

            sampleID = indexList[batchID]

            #---- Check if the ID is correctly used

            if sampleID >= 0 and sampleID < len(self.listImagePaths):

                imagePath = self.listImagePaths[sampleID]
                imageData = Image.open(imagePath).convert('RGB')

                #---- Do image transformations here

                for transformID in range (0, len(self.transformList)):
                    imageData = self.transformList[transformID].transform(imageData)

                #---- If the user did not use normalization, forcefully
                #---- do the normalization procedure
                if type(imageData) != np.ndarray:
                    trnorm = ImageTransform(ImageTransform.TRANSFORM_NORMALIZE)
                    imageData = trnorm.transform(ImageTransform.TRANSFORM_NORMALIZE, imageData)

                #---- Add new image to batch
                X[batchID, :, :, :] = imageData
                Y[batchID, :] = self.listImageLabels[sampleID]

        return X, Y

    #--------------------------------------------------------------------------------
    #---- Generates batches in a loop

    def generate (self):

        while True:
            indexList = self.generateIndexList()

            #---- Max number of batches possible to get without overlapping
            batchCount = int(len(self.listImagePaths) / self.batchsize)

            #---- We don't wana repeat the batches
            for i in range(batchCount):

                X, Y = self.generateBatch(indexList)

                yield  X,Y


#--------------------------------------------------------------------------------