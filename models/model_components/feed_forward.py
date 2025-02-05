import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config

        self.d_embed = config.d_embed
        self.d_ff = config.d_ff if hasattr(config, 'd_ff') else 4 * config.d_embed

        self.W_1 = nn.Linear(self.d_embed, self.d_ff, bias=False)
        self.W_2 = nn.Linear(self.d_ff, self.d_embed, bias=False)
        
        self.activation = nn.GELU()

        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(self.d_embed)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.W_1(x)
        x = self.activation(x)
        x = self.W_2(x)

        x = self.dropout(x)
        return x