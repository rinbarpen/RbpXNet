import torch
from torch import nn


# Use avgpool to protect smaller features
# If using maxpool, the bigger features may cover the smaller ones
# Perfer when in_channels is tiny (less than 128)
class PSPModule(nn.Module):
    def __init__(self, in_channels, out_channels, image_size):
        super(PSPModule, self).__init__()

        self.pool1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.UpsamplingBilinear2d(image_size),
        )
        self.pool2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.UpsamplingBilinear2d(image_size / 2),
        )
        self.pool3 = nn.Sequential(
            # use maxpool, ensure to conserve a part of the big features
            nn.AdaptiveMaxPool2d(4),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.UpsamplingBilinear2d(image_size / 4),
        )
        self.pool4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(6),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.UpsamplingBilinear2d(image_size / 6),
        )

    def forward(self, x):
        y1 = self.pool1(x)
        y2 = self.pool2(x)
        y3 = self.pool3(x)
        y4 = self.pool4(x)
        x = torch.cat([x, y1, y2, y3, y4], dim=1)
        return x
