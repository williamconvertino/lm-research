import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import ICLAttention
from .mlp import FMLP

class ICLBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
                
        self.attention = ICLAttention(config)
        
        self.ln_v = nn.LayerNorm(config.embed_dim_f, elementwise_affine=True)
        self.ln_qk = nn.LayerNorm(config.embed_dim_phi, elementwise_affine=True)
        
        self.dropout_attention = nn.Dropout(0.1)
        
        self.mlp = FMLP(config)
        self.ln_mlp = nn.LayerNorm(config.embed_dim_f)
        self.dropout_mlp = nn.Dropout(0.1) 
    
    def forward(self, phi, context_embeddings, f):
        
        v = self.ln_v(context_embeddings)
        q = k = self.ln_qk(phi)
        
        # Only use a residual connection on (pre-MLP) "f" when specified
        if self.config.use_f_resid:
            f = f + self.dropout_attention(self.attention(q, k, v))
            f_mlp = f + self.dropout_mlp(self.mlp(self.ln_mlp(f)))
            return f, f_mlp
        else:
            f = self.dropout_attention(self.attention(q, k, v))
            f_mlp = f + self.dropout_mlp(self.mlp(self.ln_mlp(f)))         
            return f_mlp
            