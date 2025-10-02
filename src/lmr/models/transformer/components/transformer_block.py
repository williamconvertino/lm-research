import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Attention
from .mlp import MLP

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.attention = Attention(config)
        self.ln_attention = nn.LayerNorm(config.embed_dim, elementwise_affine=True)
        self.dropout_attention = nn.Dropout(0.1)
        
        self.mlp = MLP(config)
        self.ln_mlp = nn.LayerNorm(config.embed_dim, elementwise_affine=True)
        self.dropout_mlp = nn.Dropout(0.1)
        
    def forward(self, x):
        x = x + self.dropout_attention(self.attention(self.ln_attention(x)))
        x = x + self.dropout_mlp(self.mlp(self.ln_mlp(x)))
        return x