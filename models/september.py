import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_components.attention import Attention
from .model_components.feed_forward import FeedForward

class SeptemberBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.attention = Attention(config.d_embed, config.n_heads)
        self.feed_forward = FeedForward(config.d_embed)
        
        self.attn_norm_1 = nn.LayerNorm(config.d_embed)
        self.attn_norm_2 = nn.LayerNorm(config.d_embed)
        self.ff_norm = nn.LayerNorm(config.d_embed)

    def forward(self, x):
        B, S, _ = x.shape
        k = x[:, :-1, :]
        q = v = x[:, 1:, :]
        k = self.attn_norm_1(k)
        q = self.attn_norm_2(q)
        v = self.attn_norm_2(v)
        attn_out = self.attention(q, k, v)
        x = x + torch.cat([torch.zeros(B, 1, self.config.d_embed), attn_out], dim=1)
        x = x + self.feed_forward(self.ff_norm(x))
        return x

class September(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.d_embed)
        
        self.transformer_blocks = nn.ModuleList([SeptemberBlock(config) for _ in range(config.n_layers)])

        self.ln_f = nn.LayerNorm(config.d_embed)

        self.lm_head = nn.Linear(config.d_embed, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

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

        x = self.embedding(x) # (B, S, d_embed)
        
        if self.config.center_embeddings:
            x = x - self.embedding.weight.mean(dim=0, keepdim=True)

        for block in self.transformer_blocks:
            x = block(x)
        
        x = x[:, 1:, :]
        targets = targets[:, 1:]

        x = self.ln_f(x)

        logits = self.lm_head(x)
        if targets is None:
            return logits, None
    
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.contiguous().view(-1), ignore_index=self.tokenizer.pad_id)
        return logits, loss