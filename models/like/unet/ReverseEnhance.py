import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import SwinTransformer

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # torch.div(dim_t, 2, rounding_mode='trunc')
        x1 = F.pad(x1, [torch.div(diffX, 2, rounding_mode='trunc'),
                        torch.div(diffX - diffX, 2, rounding_mode='trunc'),
                        torch.div(diffY, 2, rounding_mode='trunc'),
                        torch.div(diffY - diffY, 2, rounding_mode='trunc')])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class BackUp(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear):
        super(BackUp, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels) # DWConv, ACNet
        self.norm = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.sigmoid(x)
        return x

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.back_up1 = BackUp(1024, 512, bilinear)
        self.back_up2 = BackUp(512, 256, bilinear)
        self.back_up3 = BackUp(256, 128, bilinear)
        self.back_up4 = BackUp(128, 64, bilinear)

        self.out1 = OutConv(1024, n_classes)
        self.out2 = OutConv(512, n_classes)
        self.out3 = OutConv(256, n_classes)
        self.out4 = OutConv(128, n_classes)
        self.out5 = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        y1 = x5 = self.down4(x4)
        y2 = self.up1(x5, x4)
        y3 = self.up2(x, x3)
        y4 = self.up3(x, x2)
        y5 = self.up4(x, x1)
        
        a = self.back_up1(y1)
        y2 = y2 * a + y2
        a = self.back_up2(y2)
        y3 = y3 * a + y3
        a = self.back_up3(y3)
        y4 = y4 * a + y4
        a = self.back_up4(y4)
        y5 = y5 * a + y5
        
        y1, y2, y3, y4, y5 = self.out1(y1), self.out2(y2), self.out3(y3), self.out4(y4), self.out5(y5)
        return y1, y2, y3, y4, y5
