import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import sys


sys.path.append('../')
from network.NetStyleTransfer import StyleTransferNet

class AlgorithmStyleTransfer():

    def __init__(self):
        imsize = 512

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loader = transforms.Compose([transforms.Resize(imsize), transforms.CenterCrop(imsize), transforms.ToTensor()])
        self.unloader = transforms.ToPILImage()

    def __image_loader__(self, image_name):
        image = Image.open(image_name)
        # fake batch dimension required to fit network's input dimensions
        image = self.loader(image).unsqueeze(0)
        return image.to(self.device, torch.float)


    def run(self, pathImgContent, pathImgStyle, pathImgInput, style_weight=8000, content_weight=1, num_steps = 300):

        imageContent = self.__image_loader__(pathImgContent)
        imageStyle = self.__image_loader__(pathImgStyle)
        imageInput = self.__image_loader__(pathImgInput)

        model, style_losses, content_losses = StyleTransferNet.get(imageContent, imageStyle)
        optimizer = optim.LBFGS([imageInput.requires_grad_()])

        print('Optimizing..')
        run = [0]
        while run[0] <= num_steps:

            def closure():
                # correct the values of updated input image
                imageInput.data.clamp_(0, 1)

                optimizer.zero_grad()
                model(imageInput)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()

                return style_score + content_score

            optimizer.step(closure)

        # a last correction...
        imageInput.data.clamp_(0, 1)

        image = imageInput.cpu().clone()  # we clone the tensor to not do changes on it
        image = image.squeeze(0)  # remove the fake batch dimension
        image = self.unloader(image)

        return image
