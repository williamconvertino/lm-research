import torch
import torch.nn as nn

class ZeroMeanEmbedding(nn.Module):
    def __init__(self, vocab_size, d_embed):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_embed)
    
    def tie_weights(self, other):
        self.embedding.weight = other.weight

    def enforce_zero_mean(self):
        with torch.no_grad():
            self.embedding.weight -= self.embedding.weight.mean(dim=0, keepdim=True)

    def forward(self, x):
        if self.training: # Only need to update during training
            self.enforce_zero_mean()
        return self.embedding(x)
