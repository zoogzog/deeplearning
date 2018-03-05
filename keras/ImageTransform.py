import random
import numpy as np
from PIL import Image

#--------------------------------------------------------------------------------
class ImageTransform ():

    TRANSFORM_RESIZE = 0
    TRANSFORM_FLIP_HORIZONTAL = 1
    TRANSFORM_FLIP_VERTICAL = 2
    TRANSFORM_CROP_CENTER = 3
    TRANSFORM_NORMALIZE = 10

    #--------------------------------------------------------------------------------
    #---- Creates a transformation that can be applied to the PIL image
    #---- Transform parameters for each transform type should be the following
    #---- TRANSFORM_RESIZE: [image-width, image-height]
    #---- TRANSFORM_FLIP_HORIZONTAL: n/a
    #---- TRANSFORM_FLIP_VERTICAL: n/a

    def __init__(self, transformType, transformParameters = None):

        self.transformType = transformType
        self.transformParameters = transformParameters

    #--------------------------------------------------------------------------------
    #---- Transform the input image according to the transform type
    #---- img - a PIL image

    def transform (self, img):

        #---- Switcher
        if self.transformType == self.TRANSFORM_RESIZE:
            return self.transformResize(img, self.transformParameters[0], self.transformParameters[1])

        if self.transformType == self.TRANSFORM_FLIP_HORIZONTAL:
            return  self.transformFlipH(img)

        if self.transformType == self.TRANSFORM_FLIP_VERTICAL:
            return  self.transformFlipV(img)

        if self.transformType == self.TRANSFORM_CROP_CENTER:
            return  self.transformCropCenter(img, self.transformParameters[0], self.transformParameters[1])

        if self.transformType == self.TRANSFORM_NORMALIZE:
            return  self.transformNormalize(img)

        return img

    #--------------------------------------------------------------------------------
    #---- Scales image to fit the desired resolution
    #---- img - a PIL image
    #---- imgWidth - desired width
    #---- imgHeight - desired height
    #---- Returns: resized PIL image - if input is a PIL image
    #---- Returns: input data - if input is not a PIL image

    def transformResize (self, img, imgWidth, imgHeight):
        if type(img) != Image.Image: return img
        return img.resize((imgWidth, imgHeight))

    #---- Flips the image horizontally with 50% chance
    #---- img - a PIL image
    #---- Returns: flipped PIL image - if input is a PIL image
    #---- Returns: input data - if input is not a PIL image

    def transformFlipH (self, img):
        if type(img) != Image.Image: return img

        probability = random.uniform(0, 1)

        if probability >= 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            return img

    #---- Flips the image vertically with 50% chance
    #---- img - a PIL image
    #---- Returns: flipped PIL image - if input is a PIL image
    #---- Returns: input data - if input is not a PIL image

    def transformFlipV (self, img):
        if type(img) != Image.Image: return img

        probability = random.uniform(0, 1)

        if probability >= 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        else:
            return img

    #---- Crops the input image (center crop)
    #---- img - a PIL image
    #---- imgWidth - width of the cropped image
    #---- imgHeight - height of the cropped image
    #---- Returns: cropped PIL image - if input is a PIL image
    #---- Returns: resized PIL image - if specified w & h are too big
    #---- Returns: input data - if input is not a PIL image

    def transformCropCenter (self, img, imgWidth, imgHeight):
        w, h = img.size

        if (imgWidth >= w) or (imgHeight >= h): return img.resize((imgWidth, imgHeight))

        offsetX = int((w - imgWidth) / 2)
        offsetY = int((h - imgHeight) / 2)

        return img.crop((offsetX, offsetY, offsetX + imgWidth, offsetY + imgHeight))

    # --------------------------------------------------------------------------------


    #---- Normalizes the input image with image net coefficients
    #---- Important: This transformation outputs a numpy array
    #---- img - a PIL image
    #---- Returns: a NUMPY array

    def transformNormalize (self, img):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        imgData = np.array(img).astype(float)
        imgData = imgData / 255

        imgData[:, :, 0] = (imgData[:, :, 0] - mean[0]) / std[0]
        imgData[:, :, 1] = (imgData[:, :, 1] - mean[1]) / std[1]
        imgData[:, :, 2] = (imgData[:, :, 2] - mean[2]) / std[2]

        return imgData

# --------------------------------------------------------------------------------