import torch
import os
from .sae import SparseAutoencoder

class DictionaryLearning:
    
    def __init__(self, model, splits, max_samples=None, chunk_size=32):
        self.model = model
        self.splits = splits
        self.max_samples = max_samples
        self.chunk_size = chunk_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dl_dir = f"data/dictionary_learning/{model.config.name}"
    
    def generate_neuron_files(self, split):
        
        save_path = os.path.join(self.dl_dir, split)
        if os.path.exists(save_path):
            print(f"Neuron files already generated for {split}. Loading from {save_path}")
            return
        
        print(f"Generating neuron files for {split}...")
        os.makedirs(save_path, exist_ok=True)

        dataset = self.splits[split]
        chunk = []

        self.model.config.gather_neurons = True

        self.model.eval()
        self.model.to(self.device)
        
        with torch.no_grad():
            for i, batch in enumerate(dataset):
                if self.max_samples and i >= self.max_samples:
                    break
                
                batch = batch.to(self.device)
                _, _ = self.model(batch)
                
                neurons = self.model.get_neurons() # [{attn: neurons, ff: neurons}, ... ]
                neurons = {
                    'input': batch.cpu().numpy(),
                    'attn': [neuron.cpu().numpy() for neuron in neurons[0]['attn']],
                    'ff': [neuron.cpu().numpy() for neuron in neurons[0]['ff']],
                }
                chunk.append(neurons)
                if len(chunk) >= self.chunk_size:
                    torch.save(chunk, os.path.join(save_path, f"neurons_{i // self.chunk_size}.pt"))
                    chunk = []
                if i % 100 == 0:
                    print(f"\rGenerated neuron data for batch {i}", end="")
            if chunk:
                torch.save(chunk, os.path.join(save_path, f"neurons_{i // self.chunk_size}.pt"))
        
        self.model.to("cpu")
        self.model.config.gather_neurons = False
            
    def collect_data(self):
        print("Collecting neurons for dictionary learning...")
        for split in ["train", "test"]:
            self.generate_neuron_files(split)

    def train_sae(self, layer=0, sublayer='ff', epochs=50, sparsity_lambda = 1e-3, lr = 1e-3):
        
        save_path = os.path.join(self.dl_dir, "train")

        if os.path.exists(f"{save_path}/sae_model_{layer}_{sublayer}.pt"):
            print(f"Model already trained.")
            return
        
        print(f"No model found at {save_path}/sae_model_{layer}_{sublayer}.pt, training...")
        
        model = SparseAutoencoder(self.model.config, sparsity_lambda)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best_loss = float('inf')

        num_chunks = len(os.listdir(save_path))  # Number of neuron files

        for epoch in range(epochs):
            total_loss = 0.0
            for i in range(num_chunks):

                chunk = torch.load(f"{save_path}/neurons_{i}.pt", weights_only=False)

                for neurons in chunk:
                    batch = torch.tensor(neurons[sublayer][layer]).float().to(device)

                    optimizer.zero_grad()
                    x_hat, z = model(batch)
                    loss, recon_loss, sparsity_loss = model.loss(batch, x_hat, z)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
            if total_loss < best_loss:
                best_loss = total_loss
                torch.save(model.state_dict(), f"{self.dl_dir}/sae_model_{layer}_{sublayer}.pt")
                
            print(f"\rEpoch {epoch+1}: Loss={total_loss:.4f}", end="")

    def eval_sae(self, layer=0, sublayer='ff'):
        model = SparseAutoencoder(self.model.config)
        model.load_state_dict(torch.load(f"{self.dl_dir}/sae_model_{layer}_{sublayer}.pt"), weights_only=False)
        model.eval()
        model.to(self.device)

        save_path = os.path.join(self.dl_dir, "test")
        chunk_files = [
            f for f in os.listdir(save_path) 
            if f.startswith("neurons_") and f.endswith(".pt")
        ]
        
        # Accumulators
        total_examples = 0
        l1_sum = 0.0
        true_sparsity_sum = 0.0
        active_features_sum = 0.0
        feature_usage_total = None
        all_topk_indices = []

        for file in chunk_files:
            chunk = torch.load(os.path.join(save_path, file), weights_only=False)
            for neurons in chunk:
                batch = torch.tensor(neurons[sublayer][layer]).float().to(model.device)
                stats = model.compute_statistics(batch)

                batch_size = batch.size(0)
                total_examples += batch_size

                l1_sum += stats["l1_per_example"].sum().item()
                true_sparsity_sum += stats["avg_true_sparsity"] * batch_size
                active_features_sum += stats["active_features_per_example"].sum().item()

                if feature_usage_total is None:
                    feature_usage_total = stats["feature_usage_frequency"] * batch_size
                else:
                    feature_usage_total += stats["feature_usage_frequency"] * batch_size

                all_topk_indices.append(stats["topk_indices"])

        # Final Aggregates
        avg_l1 = l1_sum / total_examples
        avg_true_sparsity = true_sparsity_sum / total_examples
        avg_active_features = active_features_sum / total_examples
        feature_usage_freq = feature_usage_total / total_examples
        topk_indices = torch.cat(all_topk_indices, dim=0)

        # Print summary
        print("\n=== Aggregated Sparse Autoencoder Statistics ===")
        print(f"Avg L1 sparsity:            {avg_l1:.4f}")
        print(f"Avg true sparsity:          {avg_true_sparsity * 100:.2f}%")
        print(f"Avg active features/sample: {avg_active_features:.2f}")
        print(f"Min feature usage freq:     {feature_usage_freq.min().item():.4f}")
        print(f"Max feature usage freq:     {feature_usage_freq.max().item():.4f}")
        print(f"Mean feature usage freq:    {feature_usage_freq.mean().item():.4f}")
        print(f"Features used >5% of time:  {(feature_usage_freq > 0.05).sum().item()} / {len(feature_usage_freq)}")

        print(f"\n--- Top-5 Active Features (first 5 examples) ---")
        for i in range(min(5, topk_indices.shape[0])):
            print(f"Input {i}: {topk_indices[i].tolist()}")
        print("===============================================")

        # Return stats as a dictionary
        return {
            "avg_l1_sparsity": avg_l1,
            "avg_true_sparsity": avg_true_sparsity,
            "avg_active_features": avg_active_features,
            "feature_usage_frequency": feature_usage_freq,
            "topk_indices": topk_indices
        }
