import torch
import torch.nn as nn
import torchvision

# --------------------------------------------------------------------------------
# ---- Collection of neural network classes
# --------------------------------------------------------------------------------

ACTIVATION_SIGMOID = 0
ACTIVATION_SOFTMAX = 1
ACTIVATION_NONE = 2

# --------------------------------------------------------------------------------

class DenseNet121(nn.Module):

    def __init__(self, classCount, isTrained, activation):
        """
        Initialize the densenet121 network
        :param classCount: dimension of the output vector
        :param isTrained: if True then loads pre-trained weights
        :param activation: type of the activation function: ACTIVATION_SIGMOID, ACTIVATION_SOFTMAX
        """

        super(DenseNet121, self).__init__()

        self.densenet121 = torchvision.models.densenet121(pretrained=isTrained)

        kernelCount = self.densenet121.classifier.in_features

        if (activation == ACTIVATION_SIGMOID):
            self.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
        elif (activation == ACTIVATION_SOFTMAX):
            self.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Softmax())
        else:
            self.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount, classCount))

    def forward(self, x):
        x = self.densenet121(x)
        return x

    def getmodel(self):
        return self.densenet121

# --------------------------------------------------------------------------------

class DenseNet169(nn.Module):

    def __init__(self, classCount, isTrained, activation):
        """
        Initialize the densenet169 network
        :param classCount: dimension of the output vector
        :param isTrained: if True then loads pre-trained weights
        :param activation: type of the activation function: ACTIVATION_SIGMOID, ACTIVATION_SOFTMAX
        """
        super(DenseNet169, self).__init__()

        self.densenet169 = torchvision.models.densenet169(pretrained=isTrained)

        kernelCount = self.densenet169.classifier.in_features

        if (activation == ACTIVATION_SIGMOID):
            self.densenet169.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
        elif (activation == ACTIVATION_SOFTMAX):
            self.densenet169.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Softmax())
        else:
            self.densenet169.classifier = nn.Sequential(nn.Linear(kernelCount, classCount))

    def forward(self, x):
        x = self.densenet169(x)
        return x

# --------------------------------------------------------------------------------

class DenseNet201(nn.Module):

    def __init__(self, classCount, isTrained, activation):
        """
        Initialize the densenet201 network
        :param classCount: dimension of the output vector
        :param isTrained: if True then loads pre-trained weights
        :param activation: type of the activation function: ACTIVATION_SIGMOID, ACTIVATION_SOFTMAX
        """

        super(DenseNet201, self).__init__()

        self.densenet201 = torchvision.models.densenet201(pretrained=isTrained)

        kernelCount = self.densenet201.classifier.in_features

        if (activation == ACTIVATION_SIGMOID):
            self.densenet201.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
        elif (activation == ACTIVATION_SOFTMAX):
            self.densenet201.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Softmax())
        else:
            self.densenet201.classifier = nn.Sequential(nn.Linear(kernelCount, classCount))

    def forward(self, x):
        x = self.densenet201(x)
        return x

# --------------------------------------------------------------------------------

class AlexNet(nn.Module):

    def __init__(self, classCount, isTrained, activation):
        """
        Initialize the alexnet network
        :param classCount: dimension of the output vector
        :param isTrained: if True then loads pre-trained weights
        :param activation: type of the activation function: ACTIVATION_SIGMOID, ACTIVATION_SOFTMAX
        """

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

# --------------------------------------------------------------------------------

class ResNet50(nn.Module):

    def __init__(self, classCount, isTrained, activation):
        """
        Initialize the resnet50 network
        :param classCount: dimension of the output vector
        :param isTrained: if True then loads pre-trained weights
        :param activation: type of the activation function: ACTIVATION_SIGMOID, ACTIVATION_SOFTMAX
        """

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

# --------------------------------------------------------------------------------

class ResNet101(nn.Module):

    def __init__(self, classCount, isTrained, activation):
        """
        Initialize the resnet101 network
        :param classCount: dimension of the output vector
        :param isTrained: if True then loads pre-trained weights
        :param activation: type of the activation function: ACTIVATION_SIGMOID, ACTIVATION_SOFTMAX
        """

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

# --------------------------------------------------------------------------------

class Inception(nn.Module):

    def __init__(self, classCount, isTrained, activation):
        """
        Initialize the Inception network
        :param classCount: dimension of the output vector
        :param isTrained: if True then loads pre-trained weights
        :param activation: type of the activation function: ACTIVATION_SIGMOID, ACTIVATION_SOFTMAX
        """
        super(Inception, self).__init__()

        self.inception = torchvision.models.inception_v3(pretrained=isTrained)

        if (activation == ACTIVATION_SIGMOID):
            self.inception.fc = nn.Sequential(nn.Linear(2048, classCount), nn.Sigmoid())
        elif (activation == ACTIVATION_SOFTMAX):
            self.inception.fc = nn.Sequential(nn.Linear(2048, classCount), nn.Softmax())
        else:
            self.inception.fc = nn.Linear(2048, classCount)

    def forward(self, x):
        x = self.inception(x)
        return x

# --------------------------------------------------------------------------------

class VGGN16(nn.Module):

    def __init__(self, classCount, isTrained, activation):
        """
        Initialize the vggn16 network
        :param classCount: dimension of the output vector
        :param isTrained: if True then loads pre-trained weights
        :param activation: type of the activation function: ACTIVATION_SIGMOID, ACTIVATION_SOFTMAX
        """
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
        return x
# --------------------------------------------------------------------------------