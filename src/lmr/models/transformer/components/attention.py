import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings
from torch.nn.attention.flex_attention import flex_attention
from torch.nn.attention import sdpa_kernel, SDPBackend

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.W_q = nn.Linear(config.embed_dim, config.hidden_dim, bias=False)
        self.W_k = nn.Linear(config.embed_dim, config.hidden_dim, bias=False)
        self.W_v = nn.Linear(config.embed_dim, config.hidden_dim, bias=True)
        self.W_o = nn.Linear(config.hidden_dim, config.embed_dim, bias=True)
        
        self.attn_scale = 1 / math.sqrt(config.hidden_dim // config.n_heads)
        
        self.rotary_embeddings = RotaryPositionalEmbeddings(config.hidden_dim // config.n_heads, max_seq_len=config.max_seq_len + 10)
        
        # self.drop_attn = nn.Dropout(0.1)
        self.drop_resid = nn.Dropout(0.1)
        
        # causal_mask = torch.triu(torch.ones(config.max_seq_len + 10, config.max_seq_len + 10), diagonal=1).bool()
        
        # self.register_buffer("causal_mask", causal_mask)
    
    def forward(self, x):
        
        B, S, E = x.shape
        
        q = self.W_q(x).view(B, S, self.config.n_heads, self.config.hidden_dim // self.config.n_heads).transpose(1, 2).contiguous()
        k = self.W_k(x).view(B, S, self.config.n_heads, self.config.hidden_dim // self.config.n_heads).transpose(1, 2).contiguous()
        v = self.W_v(x).view(B, S, self.config.n_heads, self.config.hidden_dim // self.config.n_heads).transpose(1, 2).contiguous()
        
        q = self.rotary_embeddings(q)
        k = self.rotary_embeddings(k)
        
        def _score_mod(scores, b, h, i, j):
            keep = j <= i
            return torch.where(keep, scores, torch.full_like(scores, float("-inf")))

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            attn_output = flex_attention(
                q, k, v,
                score_mod=_score_mod,
                scale=None,
                enable_gqa=False
            )
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.config.hidden_dim)
        attn_output = self.W_o(attn_output)
        attn_output = self.drop_resid(attn_output)
        
        return attn_output
    