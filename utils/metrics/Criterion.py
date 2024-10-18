import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from utils.metrics.losses import dice_loss, focal_loss

from utils.utils import to_numpy, to_tensor

class CombinedLoss(nn.Module):
    def __init__(self, weights: tuple[float, ...], loss_functions):
        super(CombinedLoss, self).__init__()
        self.weights = weights
        self.loss_functions = loss_functions
    
    def forward(self, pred: np.ndarray|torch.Tensor, target: np.ndarray|torch.Tensor):
        # F.binary_cross_entropy(pred, target)
        # F.cross_entropy(pred, target)
        pred = to_numpy(pred)
        target = to_numpy(target)
        total_loss = 0.0
        for (weight, loss_fn) in zip(self.weights, self.loss_functions):
            total_loss = weight * loss_fn(pred, target)
        return total_loss
