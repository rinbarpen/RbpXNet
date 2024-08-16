from torch import nn
from torch.functional import F
import numpy as np

from .part import *


class UNetOrignal(nn.Module):
  def __init__(self, n_channels, n_classes, use_bilinear=False):
    super(UNetOrignal, self).__init__()
    self.n_channels = n_channels
    self.n_classes = n_classes
    self.use_bilinear = use_bilinear

    self.inc = DoubleConv(n_channels, 64)
    self.down1 = Down(64, 128)
    self.down2 = Down(128, 256)
    self.down3 = Down(256, 512)
    factor = 2 if use_bilinear else 1
    self.down4 = Down(512, 1024 // factor)
    self.up1   = Up(1024, 512 // factor, use_bilinear)
    self.up2   = Up(512, 256 // factor, use_bilinear)
    self.up3   = Up(256, 128 // factor, use_bilinear)
    self.up4   = Up(128, 64, use_bilinear)
    self.outc  = Out(64, n_classes)

  def forward(self, x):
    x1 = self.inc(x)
    x2 = self.down1(x1)
    x3 = self.down2(x2)
    x4 = self.down3(x3)
    x5 = self.down4(x4)
    x = self.up1(x5, x4)
    x = self.up2(x, x3)
    x = self.up3(x, x2)
    x = self.up4(x, x1)
    logits = self.outc(x)
    return logits


class UNet(nn.Module):
  def __init__(self, in_channels, n_classes, features=[64, 128, 256, 512, 1024], use_bilinear=False):
    super(UNet, self).__init__()

    assert len(features) >= 2, 'features must be at least 2'

    self.n_layer = len(features) - 1
    self.use_bilinear = use_bilinear
    self.conv1 = DoubleConv(in_channels, features[0])

    factor = 2 if use_bilinear else 1
    self.down_list = nn.ModuleList([
      *[Down(features[i], features[i + 1]) for i in range(len(features) - 2)],
      Down(features[len(features) - 2], features[len(features) - 1])
    ])
    self.up_list = nn.ModuleList([
      *[Up(features[i + 1], features[i] // factor, use_bilinear) for i in range(len(features) - 2, 0, -1)],
      Up(features[1], features[0], use_bilinear)
    ])
    self.out = Out(features[0], n_classes=n_classes)

  def forward(self, x):
    x = self.conv1(x)
    down_features = [x]
    # Apply down sampling
    for down in self.down_list:
      x = down(x)
      down_features.append(x)
    # Prepare features for up sampling
    down_features.pop()
    down_features.reverse()
    # Apply up sampling
    for i, up in enumerate(self.up_list):
      x = up(x, down_features[i])
    return self.out(x)
