import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import PhiAttention
from .mlp import PhiMLP

class PhiBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.attention = PhiAttention(config)
        self.ln_attention = nn.LayerNorm(config.embed_dim_phi, elementwise_affine=True)
        self.dropout_attention = nn.Dropout(0.1)
        
        self.mlp = PhiMLP(config)
        self.ln_mlp = nn.LayerNorm(config.embed_dim_phi, elementwise_affine=True)
        self.dropout_mlp = nn.Dropout(0.1)
        
    def forward(self, phi):
        phi = phi + self.dropout_attention(self.attention(self.ln_attention(phi)))
        phi = phi + self.dropout_mlp(self.mlp(self.ln_mlp(phi)))
        return phi