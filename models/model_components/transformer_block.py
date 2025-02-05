import torch
import torch.nn as nn
from .attention import Attention
from .feed_forward import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        
        self.attn_norm = nn.LayerNorm(config.d_embed)
        self.ff_norm = nn.LayerNorm(config.d_embed)

    def forward(self, x):
        x = x + self.attention(self.attn_norm(x))
        x = x + self.feed_forward(self.ff_norm(x))
        return x

class EXXTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, e, p):
        x = e + p
        e = e + self.attention(e, x, x)
        e = e + self.feed_forward(e)
        return e