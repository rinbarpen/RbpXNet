import torch
from torch import nn
from torch import functional as F
from typing import List, Optional, Dict, Tuple

from .PatchEmbedding import PatchEmbedding
from .PositionalEncoding import PositionalEncoding


class VisionAttention(nn.Module):
    def __init__(
        self,
        patch_size: int,
        qkv_channels: List[int],
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.3,
    ):
        super(VisionAttention, self).__init__()
        assert (
                hidden_dim % num_heads == 0
        ), "hidden_dim should be divided by num_heads"

        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.patch_embedding_q = PatchEmbedding(patch_size, in_channels=qkv_channels[0], embed_dim=hidden_dim)
        self.patch_embedding_k = PatchEmbedding(patch_size, in_channels=qkv_channels[1], embed_dim=hidden_dim)
        self.patch_embedding_v = PatchEmbedding(patch_size, in_channels=qkv_channels[2], embed_dim=hidden_dim)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # (B, N, H, W)
        # n_patches = H*W//P//P, alias to N
        q, k, v = self.__fit(q, k, v)  # (B, H, N, d)

        # for qkv, (B, H, N, d)
        attn = self.attention(q, k, v)
        return attn

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None):
        scale = self.head_dim ** -0.5

        scores = q.matmul(k.transpose(-2, -1) * scale)
        if mask is not None:
            negative_inf = -1e9
            scores = scores.masked_fill(mask == 0, negative_inf)

        attn_weights = self.softmax(scores)
        attn_weights = self.dropout(attn_weights)
        context = attn_weights.matmul(v)  # (B, H, N, D)

        # merge the head layers to a dim
        context = context.transpose(1, 2).contiguous().view(q.size(0), -1, self.hidden_dim)
        return context

    def __fit(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # (B, D, H//P, W//P) -> (B, n_patches, D) -> (B, n_patches, H, d) -> (B, H, n_patches, d)
        def f(qkv, g):
            B = qkv.size(0)
            x = g(qkv)
            x = x.transpose(0, 1)
            x = PositionalEncoding(self.hidden_dim, x.size(1))(x)
            x = x.transpose(0, 1)

            x = (x.view(B, -1, self.num_heads, self.head_dim)
                 .transpose(1, 2))
            return x
        q, k, v = f(q, self.patch_embedding_q), f(k, self.patch_embedding_k), f(v, self.patch_embedding_v)

        return q, k, v


# TODO:
class VisionLinearAttention(nn.Module):
    def __init__(
            self,
            patch_size: int,
            qkv_channels: List[int],
            hidden_dim: int,
            num_heads: int
    ):
        super(VisionLinearAttention, self).__init__()
        assert (
                hidden_dim % num_heads == 0
        ), "hidden_dim should be divided by num_heads"

        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.patch_embedding_q = PatchEmbedding(patch_size=patch_size, in_channels=qkv_channels[0],
                                                embed_dim=hidden_dim)
        self.patch_embedding_k = PatchEmbedding(patch_size=patch_size, in_channels=qkv_channels[1],
                                                embed_dim=hidden_dim)
        self.patch_embedding_v = PatchEmbedding(patch_size=patch_size, in_channels=qkv_channels[2],
                                                embed_dim=hidden_dim)

        self.q_softmax = nn.Softmax(dim=-1)
        self.k_softmax = nn.Softmax(dim=-2)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, a: Optional[torch.Tensor]):
        # (B, N, H, W)
        # n_patches = H*W//P//P, alias to N
        q, k, v = self.__fit(q, k, v)  # (B, H, N, d)

        # for qkv, (B, H, N, d)
        attn = self.attention(q, k, v)
        return attn

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  kv_mask: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        if kv_mask is not None:
            negative_inf = -1e9
            k_mask, v_mask = kv_mask
            k = k.masked_fill(k_mask == 0, negative_inf)
            v = v.masked_fill(kv_mask == 0, 0.0)

        q = self.q_softmax(q)
        k = self.k_softmax(k)

        scores = k.transpose(-2, -1).matmul(v)
        context = q.matmul(scores)  # (B, H, N, D)

        # merge the head layers to a dim
        context = context.transpose(1, 2).contiguous().view(q.size(0), -1, self.hidden_dim)
        return context

    def __fit(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        B = q.size(0)

        # (B, D, H//P, W//P) -> (B, n_patches, D) -> (B, n_patches, H, d) -> (B, H, n_patches, d)
        def f(qkv, g):
            B = qkv.size(0)
            x = g(qkv)
            x = x.transpose(0, 1)
            x = PositionalEncoding(self.hidden_dim, x.size(1))(x)
            x = x.transpose(0, 1)

            x = (x.view(B, -1, self.num_heads, self.head_dim)
                 .transpose(1, 2))
            return x
        q, k, v = f(q, self.w_q), f(k, self.w_k), f(v, self.w_v)

        return q, k, v


class VisionAgentAttention(nn.Module):
    def __init__(
        self,
        patch_size: int,
        qkva_channels: List[int],
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.3,
    ):
        super(VisionAgentAttention, self).__init__()
        assert (
                hidden_dim % num_heads == 0
        ), "hidden_dim should be divided by num_heads"

        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.patch_embedding_q = PatchEmbedding(patch_size=patch_size, in_channels=qkva_channels[0],
                                                embed_dim=hidden_dim)
        self.patch_embedding_k = PatchEmbedding(patch_size=patch_size, in_channels=qkva_channels[1],
                                                embed_dim=hidden_dim)
        self.patch_embedding_v = PatchEmbedding(patch_size=patch_size, in_channels=qkva_channels[2],
                                                embed_dim=hidden_dim)
        self.patch_embedding_a = PatchEmbedding(patch_size=patch_size, in_channels=qkva_channels[3],
                                                embed_dim=hidden_dim)

        self.k_softmax = nn.Softmax(dim=-1)
        self.q_softmax = nn.Softmax(dim=-1)
        self.k_dropout = nn.Dropout(p=dropout)
        self.q_dropout = nn.Dropout(p=dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, a: Optional[torch.Tensor]):
        # (B, N, H, W)
        # n_patches = H*W//P//P, alias to N for qkv, to n for a
        q, k, v, a = self.__fit(q, k, v, a)  # (B, H, N|n, d)

        # for a, (B, H, n, d) n can be not equal to q.N
        # for qkv, (B, H, N, d)
        attn = self.agent_attention(q, k, v, a)
        # (B, N, D)
        B, N, D = attn.shape
        H = W = int(N ** 0.5) * self.patch_size
        attn = (attn.view(B, N, self.patch_size, self.patch_size)
                .view(B, H // self.patch_size, W // self.patch_size, self.patch_size, self.patch_size)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
                .view(B, H, W))

        return attn

    def agent_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, a: torch.Tensor):
        scale = self.head_dim ** -0.5
        Q = q.matmul(a.transpose(-2, -1) * scale)
        Q = self.q_softmax(Q)
        Q = self.q_dropout(Q)
        K = a.matmul(k.transpose(-2, -1) * scale)
        K = self.k_softmax(K)
        K = self.k_dropout(K)
        V = v

        A = Q.matmul(K.matmul(V))  # (B, H, N, d)
        A = A.transpose(1, 2).contiguous().view(q.size(0), -1, self.hidden_dim)
        return A

    def __fit(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, a: torch.Tensor):
        # (B, D, H//P, W//P) -> (B, n_patches, D) -> (B, n_patches, H, d) -> (B, H, n_patches, d)
        def f(qkv, g):
            B = qkv.size(0)
            x = g(qkv)
            x = x.transpose(0, 1)
            x = PositionalEncoding(self.hidden_dim, x.size(1))(x)
            x = x.transpose(0, 1)

            x = (x.view(B, -1, self.num_heads, self.head_dim)
                 .transpose(1, 2))
            return x

        q, k, v, a = f(q, self.patch_embedding_q), f(k, self.patch_embedding_k), f(v, self.patch_embedding_v), f(a,
                                                                                                                 self.patch_embedding_a)
        return q, k, v, a


# (B, H, N, d)
# def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
#     scale = q.size(-1) ** -0.5

#     a = torch.matmul(q, k.transpose(-2, -1)) * scale
#     a = F.softmax(a)
#     a = torch.matmul(a, v)
#     return a


# def self_attention(qkv: torch.Tensor):
#     return attention(qkv, qkv, qkv)


# def linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
#     q = F.softmax(q)
#     k = F.softmax(k)

#     scores = torch.matmul(k.transpose(-2, -1), v)
#     a = torch.matmul(q, scores)
#     return a


# def agent_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, a: torch.Tensor, dropout: int = 0.3):
#     scale = q.size(-1) ** -0.5

#     K = torch.matmul(q, a.transpose(-2, -1) * scale)
#     K = F.softmax(K)
#     K = F.dropout(K, p=dropout)
#     Q = torch.matmul(k.transpose(-2, -1) * scale)
#     Q = F.softmax(Q)
#     Q = F.dropout(Q, p=dropout)
#     V = v

#     A = torch.matmul(Q, torch.matmul(K, V))
#     return A


class AgentAttention(nn.Module):
    def __init__(self, dropout: float=0.3):
        super(AgentAttention, self).__init__()

        self.softmax = nn.Softmax(dim=-3)
        self.dropout_k = nn.Dropout(p=dropout)
        self.dropout_q = nn.Dropout(p=dropout)

    def forward(self, q, k, v, a):
        # (B, H, N, D)
        scale = q.size(-1) ** -0.5

        K = torch.matmul(q, a.transpose(-2, -1)) * scale
        K = self.softmax(K)
        K = self.dropout_k(K)
        Q = torch.matmul(a, k.transpose(-2, -1)) * scale
        Q = self.softmax(Q)
        Q = self.dropout_q(Q)
        V = v

        A = torch.matmul(Q, torch.matmul(K, V))
        return A
