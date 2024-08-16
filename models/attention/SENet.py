import torch
from torch import nn

class SEBlock(nn.Module):
  def __init__(self, channels, reduction=16, use_res=True):
    super(SEBlock, self).__init__()
    self.reduction = reduction
    self.use_res = use_res
    self.gap = nn.AdaptiveAvgPool2d(1)
    self.activate = nn.Sequential(
      nn.Linear(channels, channels // reduction, bias=False),
      nn.ReLU(inplace=True),
      nn.Linear(channels // reduction, channels, bias=False),
      nn.Sigmoid(inplace=True)
    )
    
  def forward(self, x):
    B, C, _, _ = x.size()
    # squeeze
    se = self.gap(x).view(B, C, 1)
    # excitation
    y = self.activation(se).view(B, C, 1, 1)
    
    y = x * y.expand_as(x)
    
    return x + y if self.use_res else y

