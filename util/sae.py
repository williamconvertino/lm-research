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

    @torch.no_grad()
    def compute_statistics(self, x, threshold=1e-4):

        z = F.relu6(self.encoder(x))

        binary_z = (z > threshold).float()

        stats = {}

        # Per-example
        stats["active_features_per_example"] = binary_z.sum(dim=1).cpu() # Shape: [batch_size]
        stats["l1_per_example"] = z.sum(dim=1).cpu() # L1 sparsity per example

        # Global
        stats["avg_true_sparsity"] = (z < threshold).float().mean().item() # % of units near-zero
        stats["avg_l1_sparsity"] = z.sum(dim=1).mean().item() # Mean L1 norm
        stats["avg_active_features"] = binary_z.sum(dim=1).mean().item() # Avg # active per example

        # Feature-wise
        stats["feature_usage_frequency"] = binary_z.mean(dim=0).cpu() # How often each feature is active

        # Top-k indices
        topk_val, topk_idx = torch.topk(z, k=5, dim=1)  # top-5 active features per example
        stats["topk_indices"] = topk_idx.cpu()

        return stats

    @torch.no_grad()
    def print_statistics(self, x, threshold=1e-4, topk=5):
        """
        Compute and print diagnostic statistics for sparse activations.
        """
        stats = self.compute_statistics(x, threshold=threshold)
        
        print("\n=== Sparse Autoencoder Statistics ===")
        print(f"Avg L1 sparsity (sum of activations per example):     {stats['avg_l1_sparsity']:.4f}")
        print(f"Avg true sparsity (percent of near-zero activations): {stats['avg_true_sparsity'] * 100:.2f}%")
        print(f"Avg active features per example:                      {stats['avg_active_features']:.2f}")

        feature_freq = stats["feature_usage_frequency"]
        print(f"\n--- Feature Usage Frequency (across batch) ---")
        print(f"Min usage freq: {feature_freq.min().item():.4f}")
        print(f"Max usage freq: {feature_freq.max().item():.4f}")
        print(f"Mean usage freq: {feature_freq.mean().item():.4f}")
        print(f"Features used in >5% of examples: {(feature_freq > 0.05).sum().item()} / {len(feature_freq)}")

        # Optional: Show top-k active feature indices for the first few inputs
        print(f"\n--- Top-{topk} Active Features (first 5 examples) ---")
        topk_indices = stats["topk_indices"][:5]
        for i, row in enumerate(topk_indices):
            print(f"Input {i}: {row.tolist()}")

        print("========================================\n")