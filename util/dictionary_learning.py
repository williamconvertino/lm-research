import torch
import os
from .sae import SparseAutoencoder

class DictionaryLearning:
    
    def __init__(self, model, splits, k=None, chunk_size=32):
        self.model = model
        self.splits = splits
        self.k = k if k is not None else len(splits["train"])
        self.chunk_size = chunk_size
        self.k_chunks = self.k // chunk_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_path = f"data/dictionary_learning/{model.config.name}"
        
    def collect_data(self):
        
        if os.path.exists(self.save_path):
            print(f"Data already collected. Loading from {self.save_path}")
            return
        
        print(f"Collecting neuron data for {self.k} batches...")

        os.makedirs(self.save_path, exist_ok=True)
        
        self.model.eval()
        self.model.to(self.device)
        self.model.config.gather_neurons = True
        chunk = []
        with torch.no_grad():
            for i, batch in enumerate(self.splits["train"]):
                if i >= self.k:
                    break
                batch = batch.to(self.device)
                _, _ = self.model(batch)
                
                neurons = self.model.get_neurons() # [{attn: neurons, ff: neurons}, ... ]
                neurons = {
                    'input': batch[0].cpu().numpy(),
                    'attn': [neuron.cpu().numpy() for neuron in neurons[0]['attn']],
                    'ff': [neuron.cpu().numpy() for neuron in neurons[0]['ff']],
                }
                chunk.append(neurons)
                # torch.save(neurons, os.path.join(self.save_path, f"neurons_{i}.pt"))
                if len(chunk) >= self.chunk_size:
                    torch.save(chunk, os.path.join(self.save_path, f"neurons_{i // self.chunk_size}.pt"))
                    chunk = []
                if i % 100 == 0:
                    print(f"\rCollected neuron data for batch {i}", end="")
            if chunk:
                torch.save(chunk, os.path.join(self.save_path, f"neurons_{i // self.chunk_size}.pt"))

        self.model.to("cpu")
        self.model.config.gather_neurons = False

    def train_sae(self, layer=0, sublayer='ff', epochs=10, sparsity_lambda = 1e-3, lr = 1e-3):
        if os.path.exists(f"{self.save_path}/sae_model_{layer}_{sublayer}.pt"):
            print(f"Model already trained.")
            return
        model = SparseAutoencoder(self.model.config, sparsity_lambda)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best_loss = float('inf')

        for epoch in range(epochs):
            total_loss = 0.0
            for i in range(self.k_chunks):

                chunk = torch.load(f"{self.save_path}/neurons_{i}.pt")
                
                for neurons in chunk:
                    batch = torch.tensor(neurons[sublayer][layer]).float().to(device)
                    batch = batch.to(model.device)

                    optimizer.zero_grad()
                    x_hat, z = model(batch)
                    loss, recon_loss, sparsity_loss = model.loss(batch, x_hat, z)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                if total_loss < best_loss:
                    best_loss = total_loss
                    torch.save(model.state_dict(), f"{self.save_path}/sae_model_{layer}_{sublayer}.pt")
            
            print(f"\rEpoch {epoch+1}: Loss={total_loss:.4f}", end="")

    def eval_sae(self, layer=0, sublayer='ff'):
        model = SparseAutoencoder(self.model.config)
        model.load_state_dict(torch.load(f"{self.save_path}/sae_model_{layer}_{sublayer}.pt"))
        model.eval()
        model.to(self.device)

        stats = {}
        for i in range(self.k_chunks):
            chunk = torch.load(f"{self.save_path}/neurons_{i}.pt")
            for neurons in chunk:
                batch = torch.tensor(neurons[sublayer][layer]).float().to(model.device)
                stats.update(model.compute_statistics(batch))

        model.print_statistics(batch)
