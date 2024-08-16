from torch import nn

class PatchEmbedding(nn.Module):
  def __init__(self, image_size, patch_size, in_channels, embed_dim):
    super(PatchEmbedding, self).__init__()
    
    self.image_size = (image_size, image_size)
    self.patch_size = (patch_size, patch_size)
    self.num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
    
    self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
  def forward(self, x):
    B, C, H, W = x.shape
    
    assert H == self.image_size[0] and W == self.patch_size[1], \
      f'Input image size ({H} * {W}) does not match model ({self.image_size[0]} * {self.image_size[1]})'
    
    # (B, C, H, W) -> (B, D, H//P, W//P) -> (B, D, N) -> (B, N, D)
    # D=embed_dim, N=num_patches=(H//P)*(W//P)
    x = self.conv(x).flatten(2).transpose(1, 2)
    # x = x.view(B, self.num_patches, -1)
    
    return x
    