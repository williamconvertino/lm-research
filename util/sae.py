import torch
import torch.nn as nn
import torch.nn.functional as F

# input_dim = 512           # size of your MLP activations
# hidden_dim = 4096         # overcomplete
sparsity_lambda = 1e-3
lr = 1e-3
epochs = 50

class SparseAutoencoder(nn.Module):
    def __init__(self, config, sparsity_lambda=1e-3):
        super().__init__()
        self.d_hidden = config.d_embed * 8
        self.encoder = nn.Linear(config.d_embed, self.d_hidden, bias=True)
        self.decoder = nn.Linear(self.d_hidden, config.d_embed, bias=True)
        self.sparsity_lambda = sparsity_lambda

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        z = F.relu(self.encoder(x))  # Enforce non-negativity
        x_hat = self.decoder(z)
        return x_hat, z

    def loss(self, x, x_hat, z):
        recon_loss = F.mse_loss(x_hat, x)
        sparsity_loss = self.sparsity_lambda * z.abs().mean()
        return recon_loss + sparsity_loss, recon_loss, sparsity_loss

def train_sae(config, k, layer=0, sublayer='ff'):

    model = SparseAutoencoder(config, sparsity_lambda)
    model.to(config.device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        for i in range(k):

            neurons = torch.load(f"data/dictionary_learning/{config.name}/neurons_{i}.pt")
            batch = torch.tensor(neurons[sublayer][layer]).float().to(config.device)
            batch = batch.to(model.device)

            optimizer.zero_grad()
            x_hat, z = model(batch)
            loss, recon_loss, sparsity_loss = model.loss(batch, x_hat, z)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}: Loss={total_loss:.4f}")
