import torch
from torch import nn

from models.Attention import AgentAttention
from models.PatchEmbedding import PatchEmbedding
from models.PositionalEncoding import PositionalEncoding

from kan import KANLayer

class MLP(nn.Module):
    def __init__(self, channels, reduction):
        super(MLP, self).__init__()

        self.linear1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class VisionSelfAgentFormer(nn.Module):
    def __init__(self, in_channels, out_channels, d_model=256, max_len=5000, dropout=0.2,
                 use_kan_to_replace_mlp=False, **kwargs):
        super(VisionSelfAgentFormer, self).__init__()

        self.patch_embedding = PatchEmbedding(16, in_channels=in_channels, embed_dim=d_model)
        self.position_encoding = PositionalEncoding(d_model, max_len)

        self.attention = AgentAttention(dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.feed_forward = KANLayer(**kwargs) if use_kan_to_replace_mlp else MLP(out_channels, reduction=2)
        self.norm2 = nn.LayerNorm(d_model)

        self.attention2 = AgentAttention(dropout=dropout)
        self.norm12 = nn.LayerNorm(d_model)
        self.feed_forward2 = KANLayer(**kwargs) if use_kan_to_replace_mlp else MLP(out_channels, reduction=2)
        self.norm22 = nn.LayerNorm(d_model)


    def forward(self, qkv, a):
        first = self.first_block(qkv, a)
        second = self.second_block(first, a)

        return second

    def first_block(self, qkv, a):
        # First Block
        qkv, a = self.patch_embedding(qkv), self.patch_embedding(a)
        qkv, a = self.position_encoding(qkv), self.position_encoding(a)

        attn = self.attention(qkv, qkv, qkv, a)
        attn = attn + qkv
        attn = self.norm1(attn)

        output = self.feed_forward(attn)
        output = output + attn
        output = self.norm2(output)
        return output

    def second_block(self, qkv, a):
        # Second Block
        attn = self.attention2(qkv, qkv, qkv, a)
        attn = attn + qkv
        attn = self.norm12(attn)

        output = self.feed_forward2(attn)
        output = output + attn
        output = self.norm22(output)
        return output
