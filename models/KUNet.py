import torch
from torch import nn
from torch.nn.functional import F


class LSKModule(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(LSKModule, self).__init__()

    self.boost_conv = nn.Sequential(
      # 3x3
      nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
      nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
    )
    self.conv_list = nn.ModuleList([
      # 7x7
      nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, dilation=2),
      # 7x7
      nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=7, padding=3),
      # 7x7
      nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, padding=2, dilation=3),
    ])
  
  def forward(self, x):
    x = self.boost_conv(x)
    y1 = self.conv_list[0](x)
    y2 = self.conv_list[1](y1)
    y3 = self.conv_list[2](y2)

    return x + y1

class SoftmaxAttention(nn.Module):
  def __init__(self, embed_dim, dropout=0.1):
    super(SoftmaxAttention, self).__init__()
    self.embed_dim = embed_dim
    self.dropout = nn.Dropout(dropout)

  def forward(self, q, k, v):
    dim = q.shape[-1]
    y = torch.matmul(q, k)
    y = F.softmax(y * self.embed_dim ** -0.5, dim=-1)
    y = self.dropout(y)
    y = torch.matmul(y, v)
    return y

class LinearAttention(nn.Module):
  def __init__(self, dropout=0.1):
    super(LinearAttention, self).__init__()
    self.dropout = nn.Dropout(dropout)

  def forward(self, q, k, v):
    dim = q.shape[-1]
    F.softmax(q, dim=-1)
    F.softmax(k, dim=-2)
    q = q * dim ** -0.5
    context = torch.matmul(k, v)
    attn = torch.matmul(q, context)
    return attn.reshape(*q.shape)


class PositionAttention(nn.Module):
  """
  base on Softmax Attention
  """
  def __init__(self, embed_dim, num_heads, dropout=0.1):
    super(PositionAttention, self).__init__()
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B, C, H, W = x.size()
    y = torch.matmul(x, x)


class SmallAttn(nn.Module):
  def __init__(self, in_channels, out_channels):
    pass

class KUNet(nn.Module):
  def __init__(self, in_channels, n_classes, backbone):
    super(KUNet, self).__init__()

  def forward(self, x):
    pass

