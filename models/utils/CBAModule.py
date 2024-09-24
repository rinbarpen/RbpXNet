## CBAM Module

import torch
from torch import nn
from torch.functional import F

from models.utils.ECAModule import ECAModule
from models.utils.SAModule import SAModule
from models.utils.CAModule import CAModule


class CBAModule(nn.Module):
    def __init__(self, channels, reduction: int=1, use_optimization: bool=False):
        super(CBAModule, self).__init__()

        self.ca_mod = ECAModule(channels) if use_optimization else CAModule(channels, reduction)
        self.sa_mod = SAModule()

    def forward(self, x):
        x = x * self.ca_mod(x)
        x = x * self.sa_mod(x)
        return x
