import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings
from .model_components.attention import linear_attention_scores, rbf_attention_scores, softmax_attention_scores
from .model_components.feed_forward import FeedForward

class AugustAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        
        self.d_embed = config.d_embed
        self.n_heads = config.n_heads
        self.d_attn = config.d_embed
        
        self.W_q = nn.Linear(self.d_embed, self.d_attn * self.n_heads, bias=False)
        self.W_k = nn.Linear(self.d_embed, self.d_attn * self.n_heads, bias=False)
        self.W_v = nn.Linear(self.d_embed, self.d_attn * self.n_heads, bias=False)
        self.W_o = nn.Linear(self.d_attn * self.n_heads, self.d_embed, bias=False)

        self.rotary_embedding = RotaryPositionalEmbeddings(self.d_attn, config.max_seq_len)

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

    def forward(self, q, k=None, v=None):
        B, S, _ = q.shape

        if k is None:
            k = q
        if v is None:
            v = q
        
        q = self.W_q(q) # (B, S, d_attn * n_heads)
        k = self.W_k(k) # (B, S, d_attn * n_heads)
        v = self.W_v(v) # (B, S, d_attn * n_heads)

        q = q.view(B, S, self.n_heads, self.d_attn) # (B, S, n_heads, d_attn)
        k = k.view(B, S, self.n_heads, self.d_attn) # (B, S, n_heads, d_attn)
        v = v.view(B, S, self.n_heads, self.d_attn) # (B, S, n_heads, d_attn)
        
        q = self.rotary_embedding(q)
        k = self.rotary_embedding(k)

        q = q.transpose(1, 2) # (B, n_heads, S, d_attn)
        k = k.transpose(1, 2) # (B, n_heads, S, d_attn)
        v = v.transpose(1, 2) # (B, n_heads, S, d_attn)


        causal_mask = torch.triu(torch.ones(S, S), diagonal=1).bool().logical_not()
        causal_mask = causal_mask.to(q.device)

        attn_scores = self.attn_fn(q, k, self.scale, self.gamma, causal_mask) # (B, n_heads, S, S)
        attn_scores = self.attn_dropout(attn_scores)

        out = attn_scores @ v # (B, n_heads, S, d_attn)
        out = out.transpose(1, 2).contiguous().view(B, S, self.d_embed) # (B, S, d_embed)
        out = self.W_o(out) # (B, S, d_embed)
        out = self.proj_dropout(out)

        return out

class AugustTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
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
        
        self.aug_d_embed = config.d_embed // 3
        self.aug_d_f = config.d_embed - (2 * self.aug_d_embed)

        self.token_embedding = nn.Embedding(config.tokenizer.vocab_size, self.aug_d_embed)
        
        self.transformer_blocks = nn.ModuleList([AugustTransformerBlock(config) for _ in range(config.n_layers)])

        self.ln_f = nn.LayerNorm(config.d_embed)

        self.lm_head = nn.Linear(config.d_embed, config.tokenizer.vocab_size, bias=False)
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

        x = self.token_embedding(x) # (B, S, d_embed)
        
        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.ln_f(x)

        logits = self.lm_head(x)
        if targets is None:
            return logits, None
    
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.contiguous().view(-1), ignore_index=self.tokenizer.pad_id)
        return logits, loss