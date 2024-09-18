import torch
from torch import nn
from torch.functional import F

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()


    def forward(self, x):
        pass

class LSKDepthModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LSKDepthModule, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=7, dilation=3)

    def forward(self, x):
        pass
