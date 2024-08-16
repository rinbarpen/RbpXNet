import torch
from torch import nn
from torch.nn.functional import F

class SKBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_sizes = [3, 5]):
    super(SKBlock, self).__init__()
    
    self.convs = [nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=k//2) for k in kernel_sizes]
    self.gap = nn.AdaptiveAvgPool2d(1)
    self.activate = nn.Sequential(
      nn.Linear(out_channels, out_channels),
      nn.ReLU(inplace=True),
      nn.Linear(out_channels, out_channels),
      nn.Softmax(dim=1)
    )
    
  def forward(self, x):
    ys = [conv(x) for conv in self.convs]
    y_cat = torch.concat(ys, dim=1)
    
    B, C, _, _ = y_cat.size()
    chans = self.gap(y_cat).view(B, C)
    attn = self.activate(chans).view(B, C, 1, 1)
    
    out = torch.zeros_like(ys[0])
    for i in range(0, C // ys.len()):
      out += ys[i] * attn[:, i*C//ys.len():(i+1)*C//ys.len(), :, :].expand_as(ys[i])
    out = torch.sum(torch.stack([ys[i] * attn[:, i*(C // len(self.convs)):(i+1)*(C // len(self.convs)), :, :] for i in range(len(self.convs))]), dim=0)
    return out

