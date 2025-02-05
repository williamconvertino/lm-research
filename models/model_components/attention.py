import math
import torch
import torch.nn as nn

def linear_attention_scores(q, k, scale=1.0, gamma=None, causal_mask=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal_mask is not None:
        scores = scores.masked_fill(causal_mask == 0, 0)
    return scores

def rbf_attention_scores(q, k, scale=1.0, gamma=None, causal_mask=None):
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

def softmax_attention_scores(q, k, scale=1.0, gamma=None, causal_mask=None):
    scores = linear_attention_scores(q, k, scale)
    if causal_mask is not None:
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))
    return torch.softmax(scores, dim=-1)

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        
        self.d_embed = config.d_embed
        self.n_heads = config.n_heads
        self.d_attn = config.d_embed if hasattr(config, 'use_square_attention_headss') and config.use_square_attention_headss else config.d_embed // config.n_heads

        self.W_q = nn.Linear(self.d_embed, self.d_attn * self.n_heads, bias=False)
        self.W_k = nn.Linear(self.d_embed, self.d_attn * self.n_heads, bias=False)
        self.W_v = nn.Linear(self.d_embed, self.d_attn * self.n_heads, bias=False)
        self.W_o = nn.Linear(self.d_attn * self.n_heads, self.d_embed, bias=False)

        if hasattr(config, 'attn_fn') and config.attn_fn == 'linear':
            self.attn_fn = linear_attention_scores
        elif hasattr(config, 'attn_fn') and config.attn_fn == 'rbf':
            self.attn_fn = rbf_attention_scores
        else:
            self.attn_fn = softmax_attention_scores

        self.gamma = nn.Parameter(torch.ones(self.n_heads)) if hasattr(config, 'use_rbf_attention') and config.use_rbf_attention else None
        self.scale = 1.0 / math.sqrt(self.d_attn)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.proj_dropout = nn.Dropout(config.dropout)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, q, k=None, v=None):
        B, S, _ = q.shape

        if k is None:
            k = q
        if v is None:
            v = q
        
        q = self.W_q(q) # (B, S, d_attn * n_heads)
        k = self.W_k(k) # (B, S, d_attn * n_heads)
        v = self.W_v(v) # (B, S, d_attn * n_heads)

        q = q.view(B, S, self.n_heads, self.d_attn).transpose(1, 2) # (B, n_heads, S, d_attn)
        k = k.view(B, S, self.n_heads, self.d_attn).transpose(1, 2) # (B, n_heads, S, d_attn)
        v = v.view(B, S, self.n_heads, self.d_attn).transpose(1, 2) # (B, n_heads, S, d_attn)
    
        # Create causal mask
        causal_mask = torch.triu(torch.ones(S, S), diagonal=1).bool()
        causal_mask = causal_mask.to(q.device)

        attn_scores = self.attn_fn(q, k, self.scale, self.gamma, causal_mask) # (B, n_heads, S, S)
        attn_scores = self.attn_dropout(attn_scores)

        out = attn_scores @ v # (B, n_heads, S, d_attn)
        out = out.transpose(1, 2).contiguous().view(B, S, self.d_embed) # (B, S, d_embed)
        out = self.W_o(out) # (B, S, d_embed)
        out = self.proj_dropout(out)

        return out