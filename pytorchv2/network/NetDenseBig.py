import torch
import torch.nn as nn
import torchvision

import torch.nn.functional as F

# --------------------------------------------------------------------------------
# ---- Collection of neural network architectures that support any size of the
# ---- input image.
# --------------------------------------------------------------------------------

ACTIVATION_SIGMOID = 0
ACTIVATION_SOFTMAX = 1
ACTIVATION_NONE = 2

# --------------------------------------------------------------------------------

class BigDensenet121(nn.Module):

    def __init__(self, classCount, isTrained, activation):
        """
        :param classCount: dimension of the output vector / number of classes
        :param isTrained: if True then loads pre-trained weights
        :param activation: type of the activation function: ACTIVATION_SIGMOID or ACTIVATION_SOFTMAX
        """
        super(BigDensenet121, self).__init__()

        self.densenet = torchvision.models.densenet121(pretrained=isTrained)

        self.features = self.densenet.features
        self.classifier = self.densenet.classifier

        kernelCount = self.classifier.in_features

        if (activation == ACTIVATION_SIGMOID):
            self.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
        elif (activation == ACTIVATION_SOFTMAX):
            self.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Softmax())
        else:
            self.classifier = nn.Sequential(nn.Linear(kernelCount, classCount))

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=features.size(-1), stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out

# --------------------------------------------------------------------------------

class BigDenseNet169(nn.Module):

    def __init__(self, classCount, isTrained, activation):
        """
        :param classCount: dimension of the output vector / number of classes
        :param isTrained: if True then loads pre-trained weights
        :param activation: type of the activation function: ACTIVATION_SIGMOID or ACTIVATION_SOFTMAX
        """
        super(BigDensenet121, self).__init__()

        self.densenet = torchvision.models.densenet169(pretrained=isTrained)

        self.features = self.densenet.features
        self.classifier = self.densenet.classifier

        kernelCount = self.classifier.in_features

        if (activation == ACTIVATION_SIGMOID):
            self.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
        elif (activation == ACTIVATION_SOFTMAX):
            self.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Softmax())
        else:
            self.classifier = nn.Sequential(nn.Linear(kernelCount, classCount))

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=features.size(-1), stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out
