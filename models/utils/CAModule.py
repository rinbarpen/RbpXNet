import torch
from torch import nn
from torch import functional as F

class CAModule(nn.Module):
    def __init__(self, channels, reduction: int, **kwargs):
        super(CAModule, self).__init__()

        self.channels = channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid(inplace=True)

    def forward(self, x):
        scale = self.avg_pool(x)
        scale = self.fc(scale)
        scale = self.sigmoid(scale)

        return x * scale
