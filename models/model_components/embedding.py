import torch.nn as nn

class TopicalEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.center_embeddings = getattr(config, 'center_embeddings', False)
        self.embedding = nn.Embedding(config.vocab_size, config.d_embed)

    def forward(self, x):
        if self.center_embeddings:
            return self.embedding(x) - self.embedding.weight.mean(dim=0, keepdim=True)
        else:
            return self.embedding(x)