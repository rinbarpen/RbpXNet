## SpatialAttentionModule

import torch
from torch import nn
from torch.functional import F


class SAModule(nn.Module):
    def __init__(self):
        super(SAModule, self).__init__()

        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out_avg = torch.mean(x, dim=1, keepdim=True)
        out_max, _ = torch.max(x, dim=1, keepdim=True)
        o = torch.cat([out_avg, out_max], dim=1)
        o = self.sigmoid(self.conv(o))
        return o
