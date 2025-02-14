import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_components.transformer_block import TransformerBlock
from .model_components.embedding import TopicalEmbedding

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embedding = TopicalEmbedding(config)
        
        self.transformer_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])

        self.ln_f = nn.LayerNorm(config.d_embed)

        self.lm_head = nn.Linear(config.d_embed, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.embedding.weight

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
        
        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.ln_f(x)

        logits = self.lm_head(x)
        if targets is None:
            return logits, None
    
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.contiguous().view(-1), ignore_index=self.tokenizer.pad_id)
        return logits, loss