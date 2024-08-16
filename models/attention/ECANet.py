from torch import nn
from torch.nn.functional import F


class ECABlock(nn.Module):
  def __init__(self, channels, kernel_size=16, use_res = True):
    super(ECABlock, self).__init__()
    self.use_res = use_res
    self.gap = nn.AdaptiveAvgPool2d(1)
    
    self.activate = nn.Sequential(
      nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, groups=channels, bias=True),
      nn.Sigmoid(inplace=True)
    )
    
  def forward(self, x):
    B, C, _, _ = x.size()
    
    y = self.gap(x).view(B, C, 1)
    y = self.activate(y).view(B, C, 1, 1)
    
    y = x * y.expand_as(x)
    
    return x + y if self.use_res else y
