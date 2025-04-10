import torch
import os
import numpy as np
from tqdm import tqdm
from .sae import SparseAutoencoder

class DictionaryLearning:
    
    def __init__(self, model, splits, tokenizer, max_samples=None, chunk_size=32):
        self.model = model
        self.splits = splits
        self.tokenizer = tokenizer
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

    def train_sae(self, layer=0, sublayer='ff', epochs=10, sparsity_lambda = 1e-3, lr = 1e-3):
        
        save_path = os.path.join(self.dl_dir, "train")

        if os.path.exists(f"{self.dl_dir}/sae_model_{layer}_{sublayer}.pt"):
            print(f"Model already trained.")
            return
        
        print(f"No model found at {self.dl_dir}/sae_model_{layer}_{sublayer}.pt, training...")
        
        model = SparseAutoencoder(self.model.config, sparsity_lambda)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best_loss = float('inf')

        num_chunks = len(os.listdir(save_path))  # Number of neuron files

        for epoch in range(epochs):
            
            total_loss = 0.0
            total_recon_loss = 0.0
            total_sparsity_loss = 0.0
            
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
                    total_recon_loss += recon_loss.item()
                    total_sparsity_loss += sparsity_loss.item()
            
            total_loss /= num_chunks
            total_recon_loss /= num_chunks
            total_sparsity_loss /= num_chunks
            
            if total_loss < best_loss:
                best_loss = total_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'loss': total_loss,
                    'recon_loss': total_recon_loss,
                    'sparsity_loss': total_sparsity_loss
                    }, f"{self.dl_dir}/sae_model_{layer}_{sublayer}.pt")
                
            print(f"\rEpoch {epoch+1}: Loss={total_loss:.4f}, Recon Loss={total_recon_loss:.4f}, Sparsity Loss={total_sparsity_loss:.4f}", end="")

    def eval_sae(self, layer=0, sublayer='ff', epsilon=1e-3):
        model = SparseAutoencoder(self.model.config)
        
        # checkpoint = torch.load(f"{self.dl_dir}/sae_model_{layer}_{sublayer}.pt", weights_only=False)
        # print(f"Loading model from {self.dl_dir}/sae_model_{layer}_{sublayer}.pt")
        # print(f"Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}, Recon Loss: {checkpoint['recon_loss']:.4f}, Sparsity Loss: {checkpoint['sparsity_loss']:.4f}")
        # model.load_state_dict(checkpoint['model_state_dict'])
        model.load_state_dict(torch.load(f"{self.dl_dir}/sae_model_{layer}_{sublayer}.pt", weights_only=False))
        model.eval()
        model.to(self.device)

        save_path = os.path.join(self.dl_dir, "test")
        chunk_files = [
            f for f in os.listdir(save_path) 
            if f.startswith("neurons_") and f.endswith(".pt")
        ]

        total_loss = 0.0
        total_recon_loss = 0.0
        total_sparsity_loss = 0.0
        total_l1_sparsity = 0.0
        total_active_counts = None
        all_weights = []

        num_batches = 0
        feature_use = []

        for f in chunk_files:
            chunk = torch.load(os.path.join(save_path, f), weights_only=False)
            for neurons in chunk:
                batch = torch.tensor(neurons[sublayer][layer]).float().to(self.device)
                with torch.no_grad():
                    x_hat, z = model(batch)
                    loss, recon_loss, sparsity_loss = model.loss(batch, x_hat, z)

                    total_loss += loss.item()
                    total_recon_loss += recon_loss.item()
                    total_sparsity_loss += sparsity_loss.item()
                    total_l1_sparsity += z.abs().sum(dim=1).mean().item()

                    used = (z > epsilon).float()
                    feature_use.append(used.mean(dim=0).cpu().numpy())  # shape: (num_features,)
                    num_batches += 1

        feature_use = np.stack(feature_use, axis=0)  # (num_batches, num_features)
        mean_feature_use = feature_use.mean(axis=0)
        num_features = mean_feature_use.shape[0]

        weights = model.encoder.weight.data.cpu().numpy()
        all_weights = weights.reshape(-1)

        print("\n===== Evaluation Metrics =====")
        print(f"Avg Total Loss       : {total_loss / num_batches:.4f}")
        print(f"Avg Recon Loss       : {total_recon_loss / num_batches:.4f}")
        print(f"Avg Sparsity Loss    : {total_sparsity_loss / num_batches:.4f}")
        print(f"Avg L1 Sparsity      : {total_l1_sparsity / num_batches:.4f}")
        print(f"Sparsity % (>|{epsilon}|): {(mean_feature_use > 0).sum() / num_features * 100:.2f}%")
        print(f"Feature Use Mean     : {mean_feature_use.mean():.4f}")
        print(f"Feature Use Std      : {mean_feature_use.std():.4f}")
        print(f"Feature Use Min/Max  : {mean_feature_use.min():.4f} / {mean_feature_use.max():.4f}")
        print(f"Weights Min/Max      : {all_weights.min():.4f} / {all_weights.max():.4f}")
        print(f"Weights Mean/Std     : {all_weights.mean():.4f} / {all_weights.std():.4f}")
        
        used_often = (mean_feature_use > 0.05).sum()
        print(f"# Features Used >5%  : {used_often} / {num_features}")

        print("\n===== Top-5 Features =====")

        top5_features = mean_feature_use.argsort()[-5:][::-1]
    
        for i, feat in enumerate(top5_features):
            print(f"  Feature {feat}: used {mean_feature_use[feat]*100:.2f}% of the time")
        
        
        token_activation_counts = {feat: {} for feat in top5_features}
        token_activation_totals = {feat: {} for feat in top5_features}

        for f in chunk_files:
            chunk = torch.load(os.path.join(save_path, f), weights_only=False)
            for neurons in chunk:
                batch_activations = torch.tensor(neurons[sublayer][layer]).float().to(self.device)
                tokens = neurons['input']  # shape: (batch_size, seq_len)

                with torch.no_grad():
                    _, z = model(batch_activations)

                z = z.cpu().numpy()  # shape: (batch_size, num_features)

                for i in range(z.shape[0]):  # for each sequence
                    for j in range(z.shape[1]):  # for each feature
                        if j in top5_features:
                            activation = z[i, j]
                            if activation > epsilon:
                                # each i corresponds to a sequence of tokens
                                input_tokens = tokens[i].tolist()
                                for token in input_tokens:
                                    token_activation_totals[j][token] = token_activation_totals[j].get(token, 0.0) + activation
                                    token_activation_counts[j][token] = token_activation_counts[j].get(token, 0) + 1

        for feat in top5_features:
            print(f"\nTop tokens for Feature {feat} (used {mean_feature_use[feat]*100:.2f}% of the time):")
            
            token_scores = {
                token: token_activation_totals[feat][token] / token_activation_counts[feat][token]
                for token in token_activation_counts[feat]
            }

            top_tokens = sorted(token_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            for token_id, score in top_tokens:
                token_str = self.tokenizer.decode([int(token_id)]).strip()
                print(f"  Token '{token_str}' (ID: {token_id}) — Avg Activation: {score:.4f}")
