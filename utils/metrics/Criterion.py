import torch.nn as nn
import torch.nn.functional as F

from utils.metrics.losses import dice_loss
import torch.nn as nn
import torch.nn.functional as F

from utils.metrics.losses import dice_loss


class CombinedLoss(nn.Module):
    def __init__(self, weight_dice=0.5, weight_ce=0.5):
        super(CombinedLoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
    
    def forward(self, pred, target, n_classes):
        # 计算 Cross Entropy Loss
        ce_loss = F.binary_cross_entropy_with_logits(pred, target) if n_classes == 2 else F.cross_entropy(pred, target) 
        
        # 计算 Dice Loss
        dice = dice_loss(pred.numpy(), target.numpy(), n_classes)
        
        # 加权组合损失
        total_loss = self.weight_dice * dice + self.weight_ce * ce_loss
        return total_loss
