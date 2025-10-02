import torch
import torch.nn as nn
import torch.nn.functional as F

from lmr.models.lm_base import LMBase
from .components import TransformerBlock

class Transformer(LMBase):

    def __init__(self, config, train_mode=False):
        super().__init__()
        
        self.config = config
        self.train_mode = train_mode
        
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        
        self.transformer_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        
        self.ln_out = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        
        self.apply(self.init_weights)
        self.lm_head.weight = self.embedding.weight
        
    def forward(self, input_ids):
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        x = self.embedding(input_ids)
        
        for block in self.transformer_blocks:
            x = block(x)
                
        x = self.ln_out(x)
        logits = self.lm_head(x)
        
        return logits