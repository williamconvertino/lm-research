import torch
import torch.nn as nn
from .model_components.transformer_block import TransformerBlock

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_embed)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_embed)

        self.transformer_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])

        self.ln_f = nn.LayerNorm(config.d_embed)

        self.lm_head = nn.Linear(config.d_embed, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        B, S = x.shape

        e = self.token_embedding(x) # (B, S, d_embed)
        p = self.position_embedding(torch.arange(S).to(x.device)) # (S, d_embed)
        p = p.unsqueeze(0).repeat(B, 1, 1) # (B, S, d_embed)
        x = e + p # (B, S, d_embed)
        
        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.ln_f(x)

        logits = self.lm_head(x)
        if targets is None:
            return logits, None
    
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.contiguous().view(-1))
        return logits, loss