import torch
from torch import nn
from torch import functional as F


class ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(ASPPConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, padding=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ASPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPooling, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        size = x.shape[-2]
        x = self.avg_pool(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False) # type: ignore
        return x

class ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates, dropout=0.5):
        super(ASPPModule, self).__init__()

        self.convs = nn.ModuleList()
        # 1x1 conv | ASPPConv | ASPPooling
        self.convs.append(ASPPConv(in_channels, out_channels, kernel_size=1, dilation=0))
        for dilation in dilation_rates:
            self.convs.append(ASPPConv(in_channels, out_channels, kernel_size=3, dilation=dilation))
        self.convs.append(ASPPooling(in_channels, out_channels))

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        ys = [conv(x) for conv in self.convs]
        y = torch.cat(ys, dim=1)
        y = self.project(y)
        return y
