import torch
from torch import nn
from torch.nn.functional import F 
import math

class ScaleDotProductAttention(nn.Module):
  def __init__(self, temperature, attention_dropout=None):
    super().__init__()
    
    self.temperature = temperature
    self.dropout = nn.Dropout(attention_dropout) if attention_dropout is not None else None
    
  
  def forward(self, q, k, v, mask=None):
    # Q dot K
    attention = torch.matmul(q / self.temperature, k.transpose(2, 3))
    
    # use mask
    if mask is not None:
      attention = attention.masked_fill(mask == 0, -1e9)
    
    attention = F.softmax(attention, dim=-1)
    if self.dropout is not None:
      attention = self.dropout(attention)
    
    output = torch.matmul(attention, v)
    return output
  
class PositionEncoding(nn.Module):
  def __init__(self, d_model, max_position):
    super(PositionEncoding, self).__init__()
    
    self.register_buffer('pos_table', self._get_encoding_table(d_model=d_model, n_position=max_position))
  
  def _get_encoding_table(self, d_model, n_position):
    
    positions = torch.arange(0, n_position, dtype=torch.float32).unsqueeze(1)
    
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    
    sinusoid_table = torch.zeros(n_position, d_model)
    sinusoid_table[:, 0::2] = torch.sin(positions * div_term)
    sinusoid_table[:, 1::2] = torch.cos(positions * div_term)
    
    return torch.FloatTensor(sinusoid_table).unsqueeze(0) # (1, N, D)
  
  def forward(self, x):
    # (B, N, D)
    return x + self.pos_table[:, :x.size(1)].clone().detach()

class MultiHeadAttention(nn.Module):
  def __init__(self, n_head, d_model, d_k, d_v, dropout=None):
    super().__init__()
    
    self.n_head = n_head
    self.d_model = d_model
    self.d_k = d_k
    self.d_v = d_v
    
    self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
    self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
    self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
    
    self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
    
    self.attention = ScaleDotProductAttention(temperature=d_k ** 0.5)
    
    self.w_o = nn.Linear(n_head * d_v, d_model)
    self.dropout = nn.Dropout(dropout) if dropout is not None else None
    self.ln = nn.LayerNorm(d_model, eps=1e-6)
    
  def forward(self, q, k, v, mask=None):
    d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
    sz_b, len_q, len_k, len_v = q.size(0), q.size(1), q.size(1), q.size(1)
    
    residual = q

    # Pass through the pre-attention projection: b x lq x (n*dv)
    # Separate different heads: b x lq x n x dv
    q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
    k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
    v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

    # Transpose for attention dot product: b x n x lq x dv
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    if mask is not None:
      mask = mask.unsqueeze(1)   # For head axis broadcasting.

    q, attn = self.attention(q, k, v, mask=mask)

    #q (sz_b,n_head,N=len_q,d_k)
    #k (sz_b,n_head,N=len_k,d_k)
    #v (sz_b,n_head,N=len_v,d_v)

    # Transpose to move the head dimension back: b x lq x n x dv
    # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
    q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)

    #q (sz_b, len_q, n_head, N * d_k)
    # 最终的输出矩阵 Z
    q = self.fc(q)
    if self.dropout is not None:
      q = self.dropout(q)  

    # Add & Norm 层
    q += residual
    q = self.layer_norm(q) 

    return q, attn
  
class PositionWiseFeedForward(nn.Module):
  def __init__(self, d_in, d_model, dropout=None):
    super().__init__()
    
    self.w_1 = nn.Linear(d_in, d_model)
    self.w_2 = nn.Linear(d_model, d_in)
    self.dropout = nn.Dropout(dropout) if dropout is not None else None
    self.relu = nn.ReLU()
    self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
  
  def forward(self, x):
    residual = x
    
    x = self.w_2(self.relu(self.w_1(x)))
    if self.dropout is not None:
      x = self.dropout(x)
      
    # Add & Norm Layer
    x += residual
    x = self.layer_norm(x)
    
    return x

class EncoderLayer(nn.Module):
  def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=None):
    super(EncoderLayer, self).__init__()

    # MHA + Add & Norm
    self.self_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
    # FFN + Add & Norm
    self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout=dropout)
    
  def forward(self, x):
    x, encode_self_attn = self.self_attn(x, x, x)
    encode_output = self.pos_ffn(x)
    return encode_output, encode_self_attn

class DecoderLayer(nn.Module):
  def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=None):
    super(DecoderLayer, self).__init__()
    
    self.self_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
    self.encode_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
    self.pos_ffn = PositionWiseFeedForward(d_inner, d_model, dropout)
    
  def forward(self, x, encoder_output, encoder_self_attn_mask=None, decoder_attn_mask=None):
    
    x, decoder_self_attn = self.self_attn(x, x, x, mask=encoder_self_attn_mask)
    decoder_output, decoder_attn = self.encode_attn(x, encoder_output, encoder_output, mask=decoder_attn_mask)
    
    decoder_output = self.pos_ffn(decoder_output)
    
    return decoder_output, decoder_self_attn, decoder_attn

class Encoder(nn.Module):
  ''' A encoder model with self attention mechanism. '''

  def __init__(
    self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
    d_model, d_inner, pad_idx, dropout=0.1, n_position=200):

    super().__init__()

    self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
    self.position_enc = PositionEncoding(d_word_vec, n_position=n_position)
    self.dropout = nn.Dropout(p=dropout)
    self.layer_stack = nn.ModuleList([
        EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
        for _ in range(n_layers)])
    self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

  def forward(self, src_seq, src_mask, return_attns=False):
    enc_slf_attn_list = []

    # -- Forward --

    # Input Embedding + Position Embedding + Dropout + Norm
    enc_output = self.dropout(self.position_enc(self.src_word_emb(src_seq)))
    enc_output = self.layer_norm(enc_output)

    # N × Encoder Block
    for enc_layer in self.layer_stack:
      enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
      enc_slf_attn_list += [enc_slf_attn] if return_attns else []

    if return_attns:
      return enc_output, enc_slf_attn_list
    return enc_output,