import torch
from torch import nn
from typing import List


class LSKModule(nn.Module):
  def __init__(self, in_channels, out_channels, kernels: nn.ModuleList):
    self.convs = kernels
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.max_pool = nn.AdaptiveMaxPool2d(1)
    self.conv = nn.Conv2d(2, out_channels, kernel_size=1)
    self.sigmoid = nn.Sigmoid(inplace=True, dim=1)

  def forward(self, x):
    y = x
    high_x = [y]
    for conv in self.convs:
      y = conv(y)
      high_x.append(y)
    concat_x = torch.concat(high_x, dim=1)
    concat_x = torch.concat([self.avg_pool(concat_x), self.max_pool(concat_x)], dim=1)
    attn = self.sigmoid(self.conv(concat_x))
    x2 = torch.sum(high_x * attn, dim=1)
    return x2 * x  
