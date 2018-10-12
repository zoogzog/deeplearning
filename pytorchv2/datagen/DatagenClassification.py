import os
from PIL import Image

import torch
from torch.utils.data import Dataset

# --------------------------------------------------------------------------------
# This is a dataset generator class that extends the basic pytorch dataset generator
# so that instead of providing a directory with images we can provide a text file with
# a specific structure of the dataset. Each row of the text file contains path to an
# image and desired output of the network - a n-dimensional vector of int values.
# This dataset generator is suitable for classification tasks
# <row> = <image_path> <n-dim vector, elements separated with spaces>
# example: database/im001.jpg 0 0 0 0 1 0 0 0 0
# --------------------------------------------------------------------------------


class DatagenClassification (Dataset):

    # ------------------------------------ PRIVATE -----------------------------------
    def __init__(self, pathimgdir, pathdataset, transform):
        """
        :param pathimgdir: path to the directory that conatins images for training
        :param pathdataset: path to the file that has the description of the dataset
                            relative to the 'pathimgdir' directory.
        :param transform: transformations which will be carried for each image
        """
        self.listimgpaths = []
        self.listimglabels = []
        self.transform = transform

        # ---- Open the dataset file
        filedescriptor = open(pathdataset, "r")
        
        # ---- Scan the file and save into the internal class storage
        line = True
        
        while line:
                
            line = filedescriptor.readline()
            
            # --- if the line is not empty - then process it
            if line:
          
                lineitems = line.split()
                
                imagepath = os.path.join(pathimgdir, lineitems[0])
                imagelabel = lineitems[1:]
                imagelabel = [int(i) for i in imagelabel]

                self.listimgpaths.append(imagepath)
                self.listimglabels.append(imagelabel)
            
        filedescriptor.close()
    
    # --------------------------------------------------------------------------------
    
    def __getitem__(self, index):
        
        imagepath = self.listimgpaths[index]
        
        imagedata = Image.open(imagepath).convert('RGB')
        imagelabel = torch.FloatTensor(self.listimglabels[index])

        if self.transform is not None:
            imagedata = self.transform(imagedata)
        
        return imagedata, imagelabel

    # --------------------------------------------------------------------------------
    
    def __len__(self):
        
        return len(self.listimgpaths)

    # ------------------------------------ PUBLIC ------------------------------------

    def getsize(self):
        """
        Get the number of samples in the dataset
        :return: (int) - size of the dataset
        """

        return len(self.listimgpaths)

    # --------------------------------------------------------------------------------

    def getcount(self, index):
        """
        Get the number of samples with non-zero element in the target vergot in specified postion
        :param index: position of the non-zero element
        :return: (int) - count of samples with non-zero element
        """
        count = 0

        for i in range(0, len(self.listimglabels)):

            label = self.listimglabels[i]

            if index > len(label):
                return 0

            if label[index] != 0:
                count += 1

        return count

    # --------------------------------------------------------------------------------

    def getweights(self):
        """
        Get distribution of classes
        :return: (array) - distribution of classes in the dataset
        """
        dimension = len(self.listimglabels[0])

        distribution = []

        length = self.getsize()

        for i in range(0, dimension):
            distribution.append(self.getcount(i) / length)

        return distribution
