import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

#-------------------------------------------------------------------------------- 
#---- This class parsers a dataset file, which has the following format
#---- <$image-file-path $target-vector>
#---- * $image-file-path: is the path to an image in a database
#---- This path can be absolute or relative. If relative path is used, then
#---- the root directory path to image database has to be specified in the constructor.
#---- * $target-vector: is a vector of integers, separated with spaces
#---- No checks done to insure the correct data format of the input file
#-------------------------------------------------------------------------------- 

class DatasetGenerator (Dataset):
    
    #-------------------------------------------------------------------------------- 
    #---- Initialize the dataset generator
    #---- In: pathImageDirectory - path to the database with images
    #---- In: pathDatasetFile - path to the dataset file
    #---- In: transfrom - sequence of image transformations
    
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
    #---- Returns an image and its label specified by the index
    #---- In: index - index of the dataset element
    
    def __getitem__(self, index):
        
        imagePath = self.listImagePaths[index]
        
        imageData = Image.open(imagePath).convert('RGB')
        imageLabel= torch.FloatTensor(self.listImageLabels[index])
        
        if self.transform != None: imageData = self.transform(imageData)
        
        return imageData, imageLabel
        
    #-------------------------------------------------------------------------------- 
    #---- Returns length of the dataset
    
    def __len__(self):
        
        return len(self.listImagePaths)
    
 #-------------------------------------------------------------------------------- 
    