import torch
from torch import nn
from torch.nn.functional import F

class ChannelAttention(nn.Module):
  def __init__(self, in_channels, reduction=16):
    super(ChannelAttention, self).__init__()
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.max_pool = nn.AdaptiveMaxPool2d(1)
    self.activate = nn.Sequential(
      nn.Linear(in_channels, in_channels // reduction, bias=False),
      nn.ReLU(inplace=True),
      nn.Linear(in_channels // reduction, in_channels, bias=False),
    )
  
  def forward(self, x):
    avg_pool = self.avg_pool(x)
    max_pool = self.max_pool(x)
    avg_out = self.activate(avg_pool)
    max_out = self.activate(max_pool)
    out = avg_out + max_out
    return self.sigmoid(out)


class SpatialAttention(nn.Module):
  def __init__(self, kernel_size=7):
    super(SpatialAttention, self).__init__()
    self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
    self.sigmoid = nn.Sigmoid()
  
  def forward(self, x):
    avg_pool = torch.mean(x, dim=1, keepdim=True)
    max_pool, _ = torch.max(x, dim=1, keepdim=True)
    concat = torch.cat([avg_pool, max_pool], dim=1)
    out = self.conv(concat)
    return self.sigmoid(out)


class CBAMBlock(nn.Module):
  def __init__(self, in_channels, reduction=16, kernel_size=7):
    super(CBAMBlock, self).__init__()
    self.channel_attention = ChannelAttention(in_channels, reduction)
    self.spatial_attention = SpatialAttention(kernel_size)
  
  def forward(self, x):
    x = x * self.channel_attention(x)
    x = x * self.spatial_attention(x)
    return x
