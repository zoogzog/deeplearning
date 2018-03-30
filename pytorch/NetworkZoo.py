import torch
import torch.nn as nn
import torchvision

#--------------------------------------------------------------------------------

ACTIVATION_SIGMOID = 0
ACTIVATION_SOFTMAX = 1
ACTIVATION_NONE = 100

#--------------------------------------------------------------------------------
#---- Network architecture: DENSENET-121
#---- Initaizliation
#-------- classCount - output dimension (1 for binary classification)
#-------- isTrained - if True, loads weights, obtained during training on imagenet
#-------- activation - type of the last activation layer: ACTIVATION_SIGMOID, ACTIVATION_SOFTMAX

class DenseNet121(nn.Module):

    def __init__(self, classCount, isTrained, activation):

        super(DenseNet121, self).__init__()
		
        self.densenet121 = torchvision.models.densenet121(pretrained=isTrained)

        kernelCount = self.densenet121.classifier.in_features

        if (activation == ACTIVATION_SIGMOID): self.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
        elif (activation == ACTIVATION_SOFTMAX): self.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Softmax())
        else:  self.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount, classCount))

    def forward(self, x):
        x = self.densenet121(x)
        return x

#--------------------------------------------------------------------------------
#---- Network architecture: DENSENET-169
#---- Initaizliation
#-------- classCount - output dimension (1 for binary classification)
#-------- isTrained - if True, loads weights, obtained during training on imagenet
#-------- activation - type of the last activation layer: ACTIVATION_SIGMOID, ACTIVATION_SOFTMAX

class DenseNet169(nn.Module):
    
    def __init__(self, classCount, isTrained, activation):
        
        super(DenseNet169, self).__init__()
        
        self.densenet169 = torchvision.models.densenet169(pretrained=isTrained)
        
        kernelCount = self.densenet169.classifier.in_features

        if (activation == ACTIVATION_SIGMOID):  self.densenet169.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
        elif (activation == ACTIVATION_SOFTMAX): self.densenet169.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Softmax())
        else: self.densenet169.classifier = nn.Sequential(nn.Linear(kernelCount, classCount))
        
    def forward (self, x):
        x = self.densenet169(x)
        return x

#--------------------------------------------------------------------------------
#---- Network architecture: DENSENET-201
#---- Initaizliation
#-------- classCount - output dimension (1 for binary classification)
#-------- isTrained - if True, loads weights, obtained during training on imagenet
#-------- activation - type of the last activation layer: ACTIVATION_SIGMOID, ACTIVATION_SOFTMAX

class DenseNet201(nn.Module):
    
    def __init__ (self, classCount, isTrained, activation):
        
        super(DenseNet201, self).__init__()
        
        self.densenet201 = torchvision.models.densenet201(pretrained=isTrained)
        
        kernelCount = self.densenet201.classifier.in_features

        if (activation == ACTIVATION_SIGMOID): self.densenet201.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
        elif (activation == ACTIVATION_SOFTMAX): self.densenet201.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Softmax())
        else: self.densenet201.classifier = nn.Sequential(nn.Linear(kernelCount, classCount))
        
    def forward (self, x):
        x = self.densenet201(x)
        return x

#--------------------------------------------------------------------------------
#---- Network architecture: AlexNet
#---- Initaizliation
#-------- classCount - output dimension (1 for binary classification)
#-------- isTrained - if True, loads weights, obtained during training on imagenet
#-------- activation - type of the last activation layer: ACTIVATION_SIGMOID, ACTIVATION_SOFTMAX

class AlexNet (nn.Module):

    def __init__ (self, classCount, isTrained, activation):

        super(AlexNet, self).__init__()

        self.alexnet = torchvision.models.alexnet(pretrained=isTrained)

        if (activation == ACTIVATION_SIGMOID):
            self.alexnet.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, classCount),
                nn.Sigmoid()
            )
        elif (activation == ACTIVATION_SOFTMAX):
            self.alexnet.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, classCount),
                nn.Softmax
            )
        else:
            self.alexnet.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, classCount),
            )


    def forward(self, x):
        x = self.alexnet(x)
        return x

#--------------------------------------------------------------------------------
#---- Network architecture: ResNet50
#---- Initaizliation
#-------- classCount - output dimension (1 for binary classification)
#-------- isTrained - if True, loads weights, obtained during training on imagenet
#-------- activation - type of the last activation layer: ACTIVATION_SIGMOID, ACTIVATION_SOFTMAX

class ResNet50 (nn.Module):

    def __init__ (self, classCount, isTrained, activation):

        super(ResNet50, self).__init__()

        self.resnet50 = torchvision.models.resnet50(pretrained=isTrained)

        if (activation == ACTIVATION_SIGMOID):
            self.resnet50.fc = nn.Sequential(nn.Linear(512 * 4, classCount), nn.Sigmoid())
        elif (activation == ACTIVATION_SOFTMAX):
            self.resnet50.fc = nn.Sequential(nn.Linear(512 * 4, classCount), nn.Softmax())
        else:
            self.resnet50.fc = nn.Linear(512 * 4, classCount)

    def forward(self, x):
        x = self.resnet50(x)
        return x

#--------------------------------------------------------------------------------
#---- Network architecture: ResNet101
#---- Initaizliation
#-------- classCount - output dimension (1 for binary classification)
#-------- isTrained - if True, loads weights, obtained during training on imagenet
#-------- activation - type of the last activation layer: ACTIVATION_SIGMOID, ACTIVATION_SOFTMAX

class ResNet101 (nn.Module):

    def __init__ (self, classCount, isTrained, activation):

        super(ResNet101, self).__init__()

        self.resnet101 = torchvision.models.resnet101(pretrained=isTrained)

        if (activation == ACTIVATION_SIGMOID):
            self.resnet101.fc = nn.Sequential(nn.Linear(512 * 4, classCount), nn.Sigmoid())
        elif (activation == ACTIVATION_SOFTMAX):
            self.resnet101.fc = nn.Sequential(nn.Linear(512 * 4, classCount), nn.Softmax())
        else:
            self.resnet101.fc = nn.Linear(512 * 4, classCount)

    def forward(self, x):
        x = self.resnet101(x)
        return x


#--------------------------------------------------------------------------------
#---- Network architecture: Inception (GoogleNet)
#---- Initaizliation
#-------- classCount - output dimension (1 for binary classification)
#-------- isTrained - if True, loads weights, obtained during training on imagenet
#-------- activation - type of the last activation layer: ACTIVATION_SIGMOID, ACTIVATION_SOFTMAX

class Inception (nn.Module):

    def __init__(self, classCount, isTrained, activation):

        super(Inception, self).__init__()

        self.inception = torchvision.models.inception_v3(pretrained=isTrained)

        if (activation == ACTIVATION_SIGMOID):
         self.inception.fc =  nn.Sequential(nn.Linear(2048, classCount), nn.Sigmoid())
        elif (activation == ACTIVATION_SOFTMAX):
            self.inception.fc =  nn.Sequential(nn.Linear(2048, classCount), nn.Softmax())
        else:
            self.inception.fc = nn.Linear(2048, classCount)

    def forward(self, x):
        x = self.inception(x)
        return x

#--------------------------------------------------------------------------------
#---- Network architecture: VGG16 + Batch normalization
#---- Initaizliation
#-------- classCount - output dimension (1 for binary classification)
#-------- isTrained - if True, loads weights, obtained during training on imagenet
#-------- activation - type of the last activation layer: ACTIVATION_SIGMOID, ACTIVATION_SOFTMAX

class VGGN16 (nn.Module):

    def __init__(self, classCount, isTrained, activation):

        super(VGGN16, self).__init__()

        self.vgg = torchvision.models.vgg16_bn(pretrained=isTrained)

        if (activation == ACTIVATION_SIGMOID):
            self.vgg.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, classCount),
                nn.Sigmoid()
            )
        elif (activation == ACTIVATION_SOFTMAX):
            self.vgg.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, classCount),
                nn.Softmax()
            )
        else:
            self.vgg.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, classCount),
            )

    def forward(self, x):
        x = self.vgg(x)
        return  x
#--------------------------------------------------------------------------------