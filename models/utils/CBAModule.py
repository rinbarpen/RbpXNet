"""
    CBAM: Convolutional Block Attention Module
    http://arxiv.org/abs/1807.06521
"""

import torch
from torch import nn
import torch.nn.functional as F


from models.utils.ECAModule import ECAModule
from models.utils.CAModule import CAModule
from models.utils.SEModule import SEModule


class CBAModule(nn.Module):
    def __init__(self, channels, reduction: int=1, use_optimization: bool=False):
        super(CBAModule, self).__init__()

        self.ca_mod = ECAModule(channels) if use_optimization else SEModule(channels, reduction)
        self.sa_mod = CAModule()

    def forward(self, x):
        x = x * self.ca_mod(x)
        x = x * self.sa_mod(x)
        return x
