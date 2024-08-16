import torch
from torch import nn
from torch.nn.functional import F
from torchvision import models
from typing import List

class PSPModule(nn.Module):
  def __init__(self, in_channels, out_channels, pool_sizes: List[int]):
    self.pool_list = nn.ModuleList(
      nn.Sequential(
        nn.AdaptiveAvgPool2d(ps), 
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        ) for ps in pool_sizes)
    self.conv = nn.Conv2d(out_channels * (len(pool_sizes) + 1), out_channels, kernel_size=1, bias=False)

  def forward(self, x):
    size = x.size()
    x = F.relu(self.conv(torch.cat([
      x, *[F.interpolate(p(x), size=size[2:], mode='bilinear', align_corners=True) for p in self.pool_list]
    ]), dim=1))
    return x

class PSPNet(nn.Module):
  def __init__(self, n_classes, backbone=models.resnet50(pretained=True), ):
    if self.backbone is models.resnet50:
      self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Remove final fully connected layers

    self.ppm = PSPModule(in_channels=2048, out_channels=512, pool_sizes=[1,2,3,6])
    self.final_conv = nn.Conv2d(512, n_classes, kernel_size=1)


  def forward(self, x):
    x = self.backbone(x)
    x = self.ppm(x)
    x = self.final_conv(x)
    return x

