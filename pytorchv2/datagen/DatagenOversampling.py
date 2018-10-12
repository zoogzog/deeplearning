import os
import operator
from PIL import Image

import torch
from torch.utils.data import Dataset

# --------------------------------------------------------------------------------
# This is a dataset generator class that extends the basic pytorch dataset generator
# so that instead of providing a directory with images we can provide a text file with
# a specific structure of the dataset. Each row of the text file contains path to an
# image and desired output of the network - a n-dimensional vector of int values.
# <row> = <image_path> <n-dim vector, elements separated with spaces>
# example: database/im001.jpg 0 0 0 0 1 0 0 0 0
# This generator allows to perform oversampling of specific classes to address the
# issue of unbalanced data. For each undersampled class, this generator supplies
# the same number of samples as the max number of samples in a class in a loop.
# --------------------------------------------------------------------------------

class DatagenOversampling (Dataset):


    # ------------------------------------ PRIVATE -----------------------------------
    def __init__(self, pathimgdir, pathdataset, transform):
        """
        :param pathimgdir: path to the directory that conatins images for training
        :param pathdataset: path to the file that has the description of the dataset
        :param transform: transformations which will be carried for each image
        :return: returns True if initializing was succesfull, otherwise False
        """

        self.transform = transform

        # ---- This is a list where each i-th element represents the number of samples
        # ---- which are labelled as belonging to i-th class
        self.classcount = []

        # ---- Total number of samples in the dataset
        self.samplescount = 0

        # ---- This is a list that contains image paths for each sample in the dataset
        self.listimgpath = []

        # ---- This is a list that contains target output vectors for each sample in the dataset
        self.listoutput = []

        # ---- This is a list of lists, where each i-th element is a list that contains indexes
        # ---- of samples that do belong to i-th class
        self.classmap = []
        # ---- This is a list of all indexes with zero output target vector
        self.classmapzero = []

        self.dim = 0

        self.isok = True

        # ---- This table contains the ID's of the classes with non-zero count of samples
        self.mapClassTable = []

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

                # ---- Set the output dime after the first line is scanned
                # ---- Also initialize other containers if necessary
                # ---- If dimensions do not match return false
                if self.dim == 0:
                    self.dim = len(imagelabel)

                    # ---- Initialize class counters with zeros
                    # ---- Initialize the class map
                    for k in range(0, self.dim):
                        self.classcount.append(0)
                        self.classmap.append([])

                elif self.dim != len(imagelabel):
                    self.isok = False
                    return

                # <-------- Store data into internal conatiners -------->

                # <------- Store image paths into the list of paths
                self.listimgpath.append(imagepath)

                # <------- Store image labels
                self.listoutput.append(imagelabel)

                # <------- Increase the class counters, assuming that input data is in range [0-1]
                # <------- Store information about classes into the map
                sum = 0

                for k in range(0, self.dim):
                    self.classcount[k] = self.classcount[k] + imagelabel[k]

                    if imagelabel[k] == 1:
                        self.classmap[k].append(self.samplescount)

                    sum = sum + imagelabel[k]

                # ---- This is a normal vector (target vector is zero)
                if sum == 0:
                    self.classmapzero.append(self.samplescount)

                # <------- Increase the counter for the number of samples
                self.samplescount = self.samplescount + 1

        filedescriptor.close()

        # ----- Here calculate necessary values for the oversampling
        for i in range(0, len(self.classcount)):
            if self.classcount[i] > 0:
                self.mapClassTable.append(i)
        if len(self.classmapzero) > 0:
            self.mapClassTable.append(-1)

        # ---- Number of zero samples
        countzero = len(self.classmapzero)
        # ---- Maximum number of samples in a class
        countmax = max(self.classcount)

        maxsamplecount = max(countmax, countzero)

        # ---- Number of classes that contain at least one sample
        countclassnonzero = 0

        for i in range(0, len(self.classcount)):
            if self.classcount[i] > 0:
                countclassnonzero = countclassnonzero + 1

        if len(self.classmapzero) > 0:
            countclassnonzero = countclassnonzero + 1

        self.lengthtotal = maxsamplecount * countclassnonzero


        # ---- Succesfully extracted data
        self.isok = True

    # --------------------------------------------------------------------------------

    def __getitem__(self, index):
        classid = self.mapClassTable[index % len(self.mapClassTable)]
        sampleid = int(index / len(self.mapClassTable))

        index = 0

        if classid != -1:
            index = self.classmap[classid][sampleid % self.classcount[classid]]
        else:
            index = self.classmapzero[sampleid % len(self.classmapzero)]

        imagepath = self.listimgpath[index]
        imagedata = Image.open(imagepath).convert('RGB')
        imagelabel = torch.FloatTensor(self.listoutput[index])

        if self.transform is not None:
            imagedata = self.transform(imagedata)

        return imagedata, imagelabel

    def __len__(self):
        return self.lengthtotal

    # ------------------------------------ PUBLIC ------------------------------------

    def getsize(self):
        """
        Get the number of samples in the dataset
        :return: (int) - size of the dataset
        """

        return self.samplescount

    def getclasscount(self, iszero=False):
        """
        Get the number of samples for each label
        --- If the parameter iszero = True then returns the number of samples for whcih the output vector is zero
        --- If the parameter iszero = False then returns the number of samples for each class in array
        :param iszero: flag
        :return: (int/array) where each i-th element is a number of samples that has i-th label in the output
        """

        if iszero:
            return len(self.classmapzero)
        else:
            return self.classcount

    def getclassindexlist(self, index):
        """
        Get all the indexes of samples of a particular class
        ---- If index == -1 returns a list of indexes of samples for which the output vector is zero
        ---- If index > 0 returns a list of sample indexes for which the output vector contains a index-th label
        :param index: class index
        :return: (array) - id-s of samples of a specified class
        """
        if index == -1:
            return self.classmapzero
        else:
            return self.classmap[index]

    def getweights(self):
        """
        Get distribution of classes
        :return: (array) - distribution of classes in the dataset
        """

        distribution = []

        length = self.getsize()

        for i in range(0, self.dim):
            distribution.append(self.classcount[i] / length)

        return distribution