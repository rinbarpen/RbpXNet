import torch
from torch import nn

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x
