import torch
from torch import nn
from torchvision.models import VisionTransformer, SwinTransformer
 
class KANFormer(nn.Module):
    def __init__(self, image_size, patch_size, num_layers, num_heads, hidden_dim, mlp_dim, dropout, attention_dropout, num_classes):
        super(KANFormer, self).__init__()
        
        