import torch
import torch.nn as nn
import torchvision.models as models

import sys

sys.path.append('../')
from network.NetLayer import LayerNormalization
from network.NetLoss import StyleLoss
from network.NetLoss import ContentLoss


class StyleTransferNet():

    def get (content_img, style_img, device = "cuda"):

        layerscontent = ['conv_4']
        layersstyle = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

        # ---- Normalization constants
        nmean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        nstd = torch.tensor([0.229, 0.224, 0.225]).to(device)

        # ---- Load VGG19 as the base model
        cnn = models.vgg19(pretrained=True).features.to(device).eval()

        normalization = LayerNormalization(nmean, nstd).to(device)

        losscontentlist = []
        lossstylelist = []

        model = nn.Sequential(normalization)

        # ---- Add layers to the network
        i = 0
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in layerscontent:
                # ---- Add content loss:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                losscontentlist.append(content_loss)

            if name in layersstyle:
                # ---- Add style loss:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                lossstylelist.append(style_loss)


        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, lossstylelist,losscontentlist
