import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings
from torch.nn.attention.flex_attention import flex_attention
from torch.nn.attention import sdpa_kernel, SDPBackend

class ICLAttention(nn.Module):
    def __init__(self, config, use_wv_icl=True):
        super().__init__()
        
        self.config = config
        
        self.W_q = nn.Linear(config.embed_dim_phi, config.hidden_dim_f, bias=False) # We set biases according to https://arxiv.org/pdf/2302.08626
        self.W_k = nn.Linear(config.embed_dim_phi, config.hidden_dim_f, bias=False)
        self.W_v = nn.Linear(config.embed_dim_f, config.hidden_dim_f, bias=True)
        self.W_o = nn.Linear(config.hidden_dim_f, config.embed_dim_f, bias=True)
        
        self.drop_resid = nn.Dropout(0.1)
        
    def forward(self, q, k, v):
        
        B, S, E = q.shape
        
        q = self.W_q(q).view(B, S, self.config.n_heads_f, self.config.hidden_dim_f // self.config.n_heads_f).transpose(1, 2).contiguous()
        k = self.W_k(k).view(B, S, self.config.n_heads_f, self.config.hidden_dim_f // self.config.n_heads_f).transpose(1, 2).contiguous()
        v = self.W_v(v).view(B, S, self.config.n_heads_f, self.config.hidden_dim_f // self.config.n_heads_f).transpose(1, 2).contiguous()
        
        # Note the lack of rotary embeddings in ICL attention
        
        def _score_mod(scores, b, h, i, j):
            keep = (j < i) | ((i == 0) & (j == 0)) # Purely causal attention (only the starting token attends to itself)
            return torch.where(keep, scores, torch.full_like(scores, float("-inf")))

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            attn_output = flex_attention(
                q, k, v,
                score_mod=_score_mod,
                scale=None,
                enable_gqa=False
            )
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        attn_output = attn_output.view(B, S, self.config.hidden_dim_f)
            
        attn_output = self.W_o(attn_output)
        attn_output = self.drop_resid(attn_output)
        
        return attn_output
    
# Vanilla Attention, but with dimensions properly aligned
class PhiAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.W_q = nn.Linear(config.embed_dim_phi, config.hidden_dim_phi, bias=False)
        self.W_k = nn.Linear(config.embed_dim_phi, config.hidden_dim_phi, bias=False)
        self.W_v = nn.Linear(config.embed_dim_phi, config.hidden_dim_phi, bias=True)
        self.W_o = nn.Linear(config.hidden_dim_phi, config.embed_dim_phi, bias=True)
        
        self.rotary_embeddings = RotaryPositionalEmbeddings(config.hidden_dim_phi // config.n_heads_phi, max_seq_len=config.max_seq_len + 10)
        
        self.drop_resid = nn.Dropout(0.1)
        
    def forward(self, x):
        
        B, S, E = x.shape
        
        q = self.W_q(x).view(B, S, self.config.n_heads_phi, self.config.hidden_dim_phi // self.config.n_heads_phi).transpose(1, 2).contiguous()
        k = self.W_k(x).view(B, S, self.config.n_heads_phi, self.config.hidden_dim_phi // self.config.n_heads_phi).transpose(1, 2).contiguous()
        v = self.W_v(x).view(B, S, self.config.n_heads_phi, self.config.hidden_dim_phi // self.config.n_heads_phi).transpose(1, 2).contiguous()
        
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
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.config.hidden_dim_phi)
        attn_output = self.W_o(attn_output)
        attn_output = self.drop_resid(attn_output)
        
        return attn_output