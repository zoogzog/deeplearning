import torch
import torch.nn as nn
import torchvision

import torch.nn.functional as F


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class HighResNet(nn.Module):

    def __init__(self, inSize, classCount, activation):
        super(HighResNet, self).__init__()

        nnchout = 64
        nnstride = 3
        nnminmapsize = 16

        s0 = inSize / nnstride
        sX = s0
        count = 0

        while sX > nnminmapsize:
            s0 /= 2
            count += 1


        self.ssconv0 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=1, out_channels=nnchout, kernel_size=8, stride=3, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(nnchout)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.AvgPool2d(kernel_size=3, stride=nnstride, padding=3))
        ]))

        nf = nnchout
        for k in range(0, count -1):
            block = _DenseBlock(num_layers=16, num_input_features=nf, bn_size=4, growth_rate=16, drop_rate=0)
            self.ssconv0.add_module('denseblock%d' % (k + 1), block)

            if k != count - 2:
                trans = _Transition(num_input_features=nf, num_output_features=nf // 2)
                self.ssconv0.add_module('transition%d' % (i + 1), trans)