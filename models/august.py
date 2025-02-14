import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from .model_util.zero_mean_embedding import TopicalEmbedding
from .model_util.feed_forward import FeedForward
from .model_util.attention_functions import Attention

class August(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.aug_d_embed = config.d_embed // 3

        self.embedding = nn.Embedding(config.vocab_size, self.aug_d_embed)
        
        self.attention_blocks = nn.ModuleList([Attention(config) for _ in range(config.n_layers)])
        self.feed_forward_blocks = nn.ModuleList([FeedForward(config.d_embed) for _ in range(config.n_layers)])

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

        e = self.embedding(x) - self.embedding.weight.mean(dim=0, keepdim=True) # (B, S, d_embed)
        
        x = torch.cat([torch.zeros_like(e), e, e], dim=-1) # (B, S, 3 * aug_d_embed)
        
            
        for i in range(self.config.n_layers):
            v = torch.cat([x[:, :, self.aug_d_embed:2 * self.aug_d_embed], torch.zeros_like(x[:, :, :2 * self.aug_d_embed])], dim=-1)
            x = x + self.attention_blocks[i](q=x, k=x, v=v)
            
            if i < self.config.n_layers - 1:
                ff_in = torch.cat([x[:, :, :self.aug_d_embed], torch.zeros_like(x[:, :, 2 * self.aug_d_embed:]), x[:, :, 2 * self.aug_d_embed:]], dim=-1)
            else:
                ff_in = x

            x = x + self.feed_forward_blocks[i](ff_in)

        x = self.ln_f(x)
        x = x[:, :, :self.aug_d_embed]

        logits = self.lm_head(x)
        if targets is None:
            return logits, None
    
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.contiguous().view(-1), ignore_index=self.config.tokenizer.pad_id)
        return logits, loss