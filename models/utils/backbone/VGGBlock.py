import torch
from torch import nn
from timm.models.vgg import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn


class VGG11Block(nn.Module):
    def __init__(self, pretained=False, **kwargs):
        super(VGG11Block, self).__init__()

        self.vgg = vgg11_bn(pretained=pretained) if kwargs['bn'] else vgg11(pretained=pretained)

    def forward(self, x):
        x = self.vgg(x)
        return x
class VGG13Block(nn.Module):
    def __init__(self, pretained=False, **kwargs):
        super(VGG13Block, self).__init__()

        self.vgg = vgg13_bn(pretained=pretained) if kwargs['bn'] else vgg13(pretained=pretained)

    def forward(self, x):
        x = self.vgg(x)
        return x
class VGG16Block(nn.Module):
    def __init__(self, pretained=False, **kwargs):
        super(VGG16Block, self).__init__()

        self.vgg = vgg16_bn(pretained=pretained) if kwargs['bn'] else vgg16(pretained=pretained)

    def forward(self, x):
        x = self.vgg(x)
        return x
class VGG19Block(nn.Module):
    def __init__(self, pretained=False, **kwargs):
        super(VGG19Block, self).__init__()

        self.vgg = vgg19_bn(pretained=pretained) if kwargs['bn'] else vgg19(pretained=pretained)

    def forward(self, x):
        x = self.vgg(x)
        return x
