import torch
from torch import nn
from torch import functional as F

class ScaleFormer(nn.Module):
    def __init__(self, channels):
        super(ScaleFormer, self).__init__()
        
        self.conv7_1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7, padding=3),
        )
        self.conv7_2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.Conv2d(channels, channels, kernel_size=5, padding=2), 
        )
        self.conv7_3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.Conv2d(channels, channels, kernel_size=3, dilation=2, padding=2),
        )
        self.conv_last = nn.Sequential(
            nn.Conv2d(3 * channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.conv7_1(x)
        x2 = self.conv7_2(x)
        x3 = self.conv7_3(x)
        y = torch.cat([x1, x2, x3], dim=1)
        y = self.conv_last(y)
        return y
