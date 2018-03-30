import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

#-------------------------------------------------------------------------------- 

class DatasetGenerator (Dataset):
    
    #-------------------------------------------------------------------------------- 
    
    def __init__ (self, pathImageDirectory, pathDatasetFile, transform):
    
        self.listImagePaths = []
        self.listImageLabels = []
        self.transform = transform
    
        #---- Open file, get image paths and labels
    
        fileDescriptor = open(pathDatasetFile, "r")
        
        #---- get into the loop
        line = True
        
        while line:
                
            line = fileDescriptor.readline()
            
            #--- if not empty
            if line:
          
                lineItems = line.split()
                
                imagePath = os.path.join(pathImageDirectory, lineItems[0])
                imageLabel = lineItems[1:]
                imageLabel = [int(i) for i in imageLabel]
                
                self.listImagePaths.append(imagePath)
                self.listImageLabels.append(imageLabel)   
            
        fileDescriptor.close()
    
    #-------------------------------------------------------------------------------- 
    
    def __getitem__(self, index):
        
        imagePath = self.listImagePaths[index]
        
        imageData = Image.open(imagePath).convert('RGB')
        imageLabel= torch.FloatTensor(self.listImageLabels[index])
        
        if self.transform != None: imageData = self.transform(imageData)
        
        return imageData, imageLabel
        
    #-------------------------------------------------------------------------------- 
    
    def __len__(self):
        
        return len(self.listImagePaths)

    #--------------------------------------------------------------------------------
    #---- Get the number of samples in the database

    def getSize (self):

        return len(self.listImagePaths)

    #--------------------------------------------------------------------------------
    #---- Get the number of samples, which have non zero element
    #---- in the label, specified by the position

    def getCountNonZero (self, index):

        count = 0

        for i in range (0, len(self.listImageLabels)):

            label = self.listImageLabels[i]

            if (index > len(label)): return 0

            if (label[index] != 0): count += 1

        return count

    #--------------------------------------------------------------------------------
    #----- Get the distribution of positive and negative samples per each index

    def getClassDistribution (self):

        dimension = len(self.listImageLabels[0])

        distribution = []

        length = self.getSize()

        for i in range(0, dimension):
            distribution.append(self.getCountNonZero(i) / length)

        return distribution

 #-------------------------------------------------------------------------------- 
    