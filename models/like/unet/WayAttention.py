import torch
from torch import nn

from models.utils.attention.agent_swin import AgentSwinTransformer


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.second_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.first_conv(x)
        x = self.second_conv(x)
        # x = x + self.second_conv(x)
        return x

class Down(nn.Module):
    def __init__(self):
        super(Down, self).__init__()

        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.down_sample(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, use_bilinear=True):
        super(Up, self).__init__()

        if use_bilinear:
            self.up_sampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up_sampling = nn.ConvTranspose2d(in_channels//2, in_channels//2, kernel_size=2, stride=2)
    def forward(self, x):
        x = self.up_sampling(x)
        return x

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.bottleneck(x)
        return x

class OutConv(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(OutConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, n_classes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class OutConvMLP(nn.Module):
    def __init__(self, in_channels, mid_channels, n_classes):
        super(OutConvMLP, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, n_classes),
        )

    def forward(self, x):
        x = self.linear(x)
        return x

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, backbone=None, use_bilinear=True,
                 patch_size: int=16, hidden_dim: int=256, num_heads: int=8, dropout=0.2, **kwargs):
        super(UNet, self).__init__()

        self.in_channels = n_channels
        self.n_classes = n_classes
        self.conv1 = DoubleConv(n_channels, 64)
        self.down1 = Down()
        self.conv2 = DoubleConv(64, 128)
        self.down2 = Down()
        self.conv3 = DoubleConv(128, 256)
        self.down3 = Down()
        self.conv4 = DoubleConv(256, 512)
        self.down4 = Down()
        self.bottleneck = Bottleneck(512, 512)
        self.up1 = Up(512, use_bilinear=use_bilinear)
        self.conv5 = DoubleConv(1024, 256)
        self.up2 = Up(256, use_bilinear=use_bilinear)
        self.conv6 = DoubleConv(512, 128)
        self.up3 = Up(128, use_bilinear=use_bilinear)
        self.conv7 = DoubleConv(256, 64)
        self.up4 = Up(64, use_bilinear=use_bilinear)
        self.conv8 = DoubleConv(128, 64)
        self.out_conv = OutConv(64, n_classes)

        # qkv_channels_list = [[64, 64, 64],
        #                      [128, 128, 128],
        #                      [256, 256, 256],
        #                      [512, 512, 512]]
        self.agent_list = nn.ModuleList()
        # for qkv_channels in qkv_channels_list:
        self.agent_list.extend([
            AgentSwinTransformer(512, 4, 64),
            AgentSwinTransformer(256, 4, 128),
            AgentSwinTransformer(128, 4, 256),
            AgentSwinTransformer(64, 4, 512),
        ])


    def forward(self, x):
        y = x

        x1 = self.conv1(y)
        y = self.down1(x1)
        x2 = self.conv2(y)
        y = self.down2(x2)
        x3 = self.conv3(y)
        y = self.down3(x3)
        x4 = self.conv4(y)
        y = self.down4(x4)
        y = self.bottleneck(y)
        y = self.up1(y)

        x4 = x4 + self.agent_list[3](x4)
        y = torch.concat([y, x4], dim=1)
        y = self.conv5(y)
        y = self.up2(y)
        x3 = x3 + self.agent_list[2](x3)
        y = torch.concat([y, x3], dim=1)
        y = self.conv6(y)
        y = self.up3(y)
        x2 = x2 + self.agent_list[1](x2)
        y = torch.concat([y, x2], dim=1)
        y = self.conv7(y)
        y = self.up4(y)
        x1 = x1 + self.agent_list[0](x1)
        y = torch.concat([y, x1], dim=1)
        y = self.conv8(y)
        y = self.out_conv(y)
        return y
