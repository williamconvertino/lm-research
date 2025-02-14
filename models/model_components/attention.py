import math
import torch
import torch.nn as nn
from torchtune.modules import RotaryPositionalEmbeddings

def softmax_attention_scores(q, k, scale=1.0, causal_mask=None, gamma=None):
    scores = linear_attention_scores(q, k, scale)
    if causal_mask is not None:
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))
    return torch.softmax(scores, dim=-1)

def linear_attention_scores(q, k, scale=1.0, causal_mask=None, gamma=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal_mask is not None:
        scores = scores.masked_fill(causal_mask == 0, 0)
    return scores

def rbf_attention_scores(q, k, scale=1.0, causal_mask=None, gamma=None):
    q_norm = (q ** 2).sum(dim=-1, keepdim=True)
    k_norm = (k ** 2).sum(dim=-1, keepdim=True)
    qk = torch.matmul(q, k.transpose(-2, -1))
    dist = q_norm - 2 * qk + k_norm.transpose(-2, -1)
    dist = torch.clamp(dist, min=0.0)
    
    assert gamma is not None, "gamma must be provided for RBF attention"
    gamma = gamma.view(1, -1, 1, 1)

    scores = torch.exp(-gamma * scale * dist)
    if causal_mask is not None:
        scores = scores.masked_fill(causal_mask == 0, 0)
    return scores

class Attention(nn.Module):
    def __init__(self, d_embed, n_heads, attn_fn_type='softmax', max_seq_len=4096):
        super().__init__()

        self.d_embed = d_embed
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        
        self.W_q = nn.Linear(self.d_embed, self.d_embed * self.n_heads, bias=False)
        self.W_k = nn.Linear(self.d_embed, self.d_embed * self.n_heads, bias=False)
        self.W_v = nn.Linear(self.d_embed, self.d_embed * self.n_heads, bias=False)
        self.W_o = nn.Linear(self.d_embed * self.n_heads, self.d_embed, bias=False)

        self.rotary_embedding = RotaryPositionalEmbeddings(self.d_embed, self.max_seq_len)
        
        if attn_fn_type == 'linear':
            self.attn_fn = linear_attention_scores
        elif attn_fn_type == 'rbf':
            self.attn_fn = rbf_attention_scores
        else:
            self.attn_fn = softmax_attention_scores

        self.gamma = nn.Parameter(torch.ones(self.n_heads)) if attn_fn_type == 'rbf' else None
        self.scale = 1.0 / math.sqrt(self.d_embed)

        self.attn_dropout = nn.Dropout(0.1)
        self.proj_dropout = nn.Dropout(0.1)

    def forward(self, q, k=None, v=None):
        B, S, _ = q.shape

        if k is None:
            k = q
        if v is None:
            v = q
        
        q = self.W_q(q) # (B, S, d_embed * n_heads)
        k = self.W_k(k) # (B, S, d_embed * n_heads)
        v = self.W_v(v) # (B, S, d_embed * n_heads)

        q = q.view(B, S, self.n_heads, self.d_embed) # (B, S, n_heads, d_embed)
        k = k.view(B, S, self.n_heads, self.d_embed) # (B, S, n_heads, d_embed)
        v = v.view(B, S, self.n_heads, self.d_embed) # (B, S, n_heads, d_embed)
        
        q = self.rotary_embedding(q)
        k = self.rotary_embedding(k)

        q = q.transpose(1, 2) # (B, n_heads, S, d_embed)
        k = k.transpose(1, 2) # (B, n_heads, S, d_embed)
        v = v.transpose(1, 2) # (B, n_heads, S, d_embed)

        causal_mask = torch.triu(torch.ones(S, S), diagonal=1).bool().logical_not()
        causal_mask = causal_mask.to(q.device)

        attn_scores = self.attn_fn(q, k, self.scale, causal_mask, self.gamma) # (B, n_heads, S, S)
        attn_scores = self.attn_dropout(attn_scores)

        out = attn_scores @ v # (B, n_heads, S, d_embed)
        out = out.transpose(1, 2).contiguous().view(B, S, self.d_embed) # (B, S, d_embed)
        out = self.W_o(out) # (B, S, d_embed)
        out = self.proj_dropout(out)

        return out