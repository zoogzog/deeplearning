import os
from PIL import Image

import torch
from torch.utils.data import Dataset

# --------------------------------------------------------------------------------
# This is a dataset generator class that extends the basic pytorch dataset generator
# so that instead of providing a directory with images we can provide a text file with
# a specific structure of the dataset. Each row in the file contains path to the input
# file and path to the output image file.
# <row> = <input image path> <output image path>
# --------------------------------------------------------------------------------

class DatagenAutoencoder(Dataset):

    # ------------------------------------ PRIVATE -----------------------------------
    def __init__(self, pathimgdir, pathdataset, transformin, transformout):
        """
        :param pathimgdir: path to the directory that contains images
        :param pathdataset: path to the file of the dataset
        :param transformin: transform function for the input images
        :param transformout: transform functio for the output image
        """

        self.listimginput = []
        self.listimgoutput = []
        self.transformin = transformin
        self.transformout = transformout

        # ---- Open the dataset file
        filedescriptor = open(pathdataset, "r")

        # ---- Scan the file and save into the internal class storage
        line = True

        while line:

            line = filedescriptor.readline()

            # --- if the line is not empty - then process it
            if line:
                lineitems = line.split()

                imagepathin = os.path.join(pathimgdir, lineitems[0])
                imagepathout = os.path.join(pathimgdir, lineitems[1])

                self.listimginput.append(imagepathin)
                self.listimgoutput.append(imagepathout)

        filedescriptor.close()

    # --------------------------------------------------------------------------------

    def __getitem__(self, index):

        imgpathin = self.listimginput[index]
        imgpathout = self.listimgoutput[index]

        datain = Image.open(imgpathin).convert('RGB')
        dataout = Image.open(imgpathout).convert('RGB')

        datain = self.transformin(datain)
        dataout = self.transformout(dataout)

        return datain, dataout

    # --------------------------------------------------------------------------------

    def __len__(self):

        return len(self.listimginput)

    # --------------------------------------------------------------------------------

    def getsize(self):
        """
        Get the number of samples in the dataset
        :return: (int) - size of the dataset
        """

        return len(self.listimginput)

    def getweights(self):
        return None
