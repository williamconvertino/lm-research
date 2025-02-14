import math
import torch
import torch.nn as nn
from .model_components.feed_forward import FeedForward
from .model_components.attention import linear_attention_scores, rbf_attention_scores, softmax_attention_scores

class AugustAttention(nn.Module):
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
    
        causal_mask = torch.triu(torch.ones(S, S), diagonal=1).bool().logical_not()
        causal_mask = causal_mask.to(q.device)

        attn_scores = self.attn_fn(q, k, self.scale, self.gamma, causal_mask) # (B, n_heads, S, S)
        attn_scores = self.attn_dropout(attn_scores)

        out = attn_scores @ v # (B, n_heads, S, d_attn)
        out = out.transpose(1, 2).contiguous().view(B, S, self.d_embed) # (B, S, d_embed)
        out = self.W_o(out) # (B, S, d_embed)
        out = self.proj_dropout(out)

        return out

class AugustBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.sub_d_embed = config.d_embed // 3
        self.sub_d_f = config.d_embed - (2 * self.sub_d_embed)

        self.attention = AugustAttention(config)
        self.feed_forward = FeedForward(config)
        
        self.attn_norm = nn.LayerNorm(config.d_embed)
        self.ff_norm = nn.LayerNorm(config.d_embed)

    def forward(self, x):
        x = x + self.attention(self.attn_norm(x))
        x = x + self.feed_forward(self.ff_norm(x))
        return x
    
class August(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.sub_d_embed = config.d_embed // 3
        self.sub_d_f = config.d_embed - (2 * self.sub_d_embed)

        self.token_embedding = nn.Embedding(config.vocab_size, config.sub_d_embed)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_embed)

        self.transformer_blocks = nn.ModuleList([AugustBlock(config) for _ in range(config.n_layers)])

        self.ln_f = nn.LayerNorm(config.d_embed)

        self.lm_head = nn.Linear(config.d_embed, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        B, S = x.shape

        e = self.token_embedding(x) # (B, S, d_embed)
        e = torch.cat([torch.zeros(B, S, self.sub_d_f), e, e], dim=-1)
        p = self.position_embedding(torch.arange(S).to(x.device)) # (S, d_embed)
        p = p.unsqueeze(0).repeat(B, 1, 1) # (B, S, d_embed)
        x = e + p # (B, S, d_embed)
        
        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.ln_f(x)

        logits = self.lm_head(x)
        if targets is None:
            return logits, None
    
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.contiguous().view(-1), ignore_index=self.tokenizer.pad_id)
        return logits, loss