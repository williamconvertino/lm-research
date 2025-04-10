import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):
    def __init__(self, config, sparsity_lambda=1e-3):
        super().__init__()
        self.d_embed = config.d_embed

        if config.name in ["gformer", "divformer"]:
            self.d_embed = config.d_embed // 2

        self.d_hidden = config.d_embed * 8
        self.encoder = nn.Linear(self.d_embed, self.d_hidden, bias=False)
        self.decoder = nn.Linear(self.d_hidden, self.d_embed, bias=False)
        self.sparsity_lambda = sparsity_lambda

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        z = F.relu6(self.encoder(x))  # Enforce non-negativity
        x_hat = self.decoder(z)
        return x_hat, z

    def loss(self, x, x_hat, z):
        recon_loss = F.mse_loss(x_hat, x)
        sparsity_loss = self.sparsity_lambda * z.abs().sum(dim=1).mean()
        return recon_loss + sparsity_loss, recon_loss, sparsity_loss