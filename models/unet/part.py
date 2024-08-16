import torch
from torch import nn
from torch.functional import F
import numpy as np


class DoubleConv(nn.Module):
  def __init__(self, in_channels, out_channels, mid_channels=None):
    super(DoubleConv, self).__init__()
    if not mid_channels:
      mid_channels = out_channels
    self.double_conv = nn.Sequential(
      nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(mid_channels),
      nn.ReLU(inplace=True),
      nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True)
    )
    
  def forward(self, x):
    x = self.double_conv(x)
    return x
  
class Down(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(Down, self).__init__()
    self.maxpool_conv = nn.Sequential(
      nn.MaxPool2d(2),
      DoubleConv(in_channels, out_channels)
    )
  
  def forward(self, x):
    x = self.maxpool_conv(x)
    return x


class Up(nn.Module):
  def __init__(self, in_channels, out_channels, use_bilinear=False):
    super(Up, self).__init__()
    
    if use_bilinear:
      self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
      self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
    else:
      self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
      self.conv = DoubleConv(in_channels, out_channels)
  
  def forward(self, x1, x2):
    x1 = self.up(x1)

    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]

    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])

    x = torch.cat([x2, x1], dim=1)
    return self.conv(x)


class Out(nn.Module):
  def __init__(self, in_channels, n_classes):
    super(Out, self).__init__()
    
    # self.conv = DoubleConv(in_channels, in_channels)
    self.out = nn.Conv2d(in_channels, n_classes, kernel_size=1, bias=False)
    
  def forward(self, x):
    # x = self.conv(x)
    x = self.out(x)
    return x
