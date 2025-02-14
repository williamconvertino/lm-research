import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, d_embed, d_hidden=None):
        super().__init__()

        if d_hidden is None:
            d_hidden = d_embed * 4

        self.W_1 = nn.Linear(d_embed, d_hidden, bias=False)
        self.W_2 = nn.Linear(d_hidden, d_embed, bias=False)
        
        self.activation = nn.GELU()

        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(d_embed)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.W_1(x)
        x = self.activation(x)
        x = self.W_2(x)

        x = self.dropout(x)
        return x