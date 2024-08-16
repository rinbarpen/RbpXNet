import math
import torch
from torch import nn
import torch.nn.functional as F


class Attention(nn.Module):
  def __init__(self):
    super(Attention, self).__init__()
    
    self.softmax = nn.Softmax(dim=-1)
  
  def forward(self, q, k, v):
    return torch.matmul(self.softmax(torch.matmul(q, k)), v)

class SelfAttention(nn.Module):
  def __init__(self):
    super(SelfAttention, self).__init__()
    
    self.softmax = nn.Softmax(dim=-1)
  
  def forward(self, x):
    return torch.matmul(self.softmax(torch.matmul(x, x)), x)

class LinearAttention(nn.Module):
  def __init__(self):
    super(LinearAttention, self).__init__()
    

  def forward(self, q, k, v):
    return torch.bmm(q.permute(0, 2, 1), torch.bmm(k.permute(0, 2, 1), v))


class AgentAttention(nn.Module):
  def __init__(self):
    super(AgentAttention, self).__init__()

    self.softmax = nn.Softmax(dim=-1)
    
  def forward(self, q, a, k, v):
    qq = self.softmax(torch.matmul(q, a))
    kk = self.softmax(torch.matmul(a, k))
    vv = v
    # N, d
    return torch.matmul(qq, torch.matmul(kk, vv))



class ChannelAttnBlock(nn.Module):
  def __init__(self, channels, gamma=2, bias=1):
    super(ChannelAttnBlock, self).__init__()
    
    self.gamma = gamma
    self.bias = bias
    self.channels = channels
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.max_pool = nn.AdaptiveMaxPool2d(1)
    k = self._mid_kernel_size()
    self.conv = nn.Conv1d(channels * 2, channels * 2, kernel_size=k, padding=k//2, bias=False)
    
    self.sigmoid = nn.Sigmoid()
  
  def forward(self, x):
    B, C, H, W = x.size()
    attn1 = self.avg_pool(x).view(B, C, 1)
    attn2 = self.max_pool(x).view(B, C, 1) 
    attn = self.conv(torch.concat([attn1, attn2], dim=1))
    attn = attn[:, :C, :] + attn[:, C:, :]
    attn = self.sigmoid(attn)
    attn = attn.unsqueeze(-1)
    return x * attn.expand_as(x)
    
  def _mid_kernel_size(self):
    t = (math.log(self.channels, 2) + self.bias) / self.gamma
    it = int(t)
    return it+1 if it % 2 == 0 else it

class SpatialAttnBlock(nn.Module):
  def __init__(self, kernel_size=7):
    super(SpatialAttnBlock, self).__init__()
    
    self.kernel_size = kernel_size
    
    self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
    self.sigmoid = nn.Sigmoid()
    
  def forward(self, x):
    avg_pool = torch.mean(x, dim=1, keepdim=True)
    max_pool, _ = torch.max(x, dim=1, keepdim=True)
    concat = torch.cat([avg_pool, max_pool], dim=1)
    out = self.conv(concat)
    out = self.sigmoid(out)
    return x * out.expand_as(x)

class ChannelSpatialBlock(nn.Module):
  def __init__(self, channels, gamma=2, bias=1, kernel_size=7):
    super(ChannelSpatialBlock, self).__init__()
    
    self.channel_attention = ChannelAttnBlock(channels, gamma, bias)
    self.spatial_attention = SpatialAttnBlock(kernel_size)
    
  def forward(self, x):
    x = self.channel_attention(x)
    x = self.spatial_attention(x)
    return x