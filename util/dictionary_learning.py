from tqdm import tqdm

# Inside DictionaryLearning class

def train_sae(self, layer=0, sublayer='ff', epochs=10, sparsity_lambda=1e-3, lr=1e-3):
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

        print(f"\nEpoch {epoch+1}/{epochs}")
        for i in tqdm(range(num_chunks), desc="Training chunks"):
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

        print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, Recon Loss={total_recon_loss:.4f}, Sparsity Loss={total_sparsity_loss:.4f}")


def eval_sae(self, layer=0, sublayer='ff', epsilon=1e-3):
    model = SparseAutoencoder(self.model.config)
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
    num_batches = 0
    feature_use = []

    print("\nEvaluating model...")
    for f in tqdm(chunk_files, desc="Evaluating chunks"):
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
                feature_use.append(used.mean(dim=0).cpu().numpy())
                num_batches += 1

    feature_use = np.stack(feature_use, axis=0)
    mean_feature_use = feature_use.mean(axis=0)
    num_features = mean_feature_use.shape[0]

    weights = model.encoder.weight.data.cpu().numpy()
    all_weights = weights.reshape(-1)

    print("\n===== Evaluation Metrics =====")
    print(f"Avg Total Loss       : {total_loss / num_batches:.4f}")
    print(f"Avg Recon Loss       : {total_recon_loss / num_batches:.4f}")
    print(f"Avg Sparsity Loss    : {total_sparsity_loss / num_batches:.4f}")
    print(f"Avg L1 Sparsity      : {total_l1_sparsity / num_batches:.4f}")

    used_often = (mean_feature_use > 0.05).sum()
    print(f"# Features Used >5%  : {used_often} / {num_features}")

    print("\n===== Top-5 Features =====")
    top5_features = mean_feature_use.argsort()[-5:][::-1]
    for i, feat in enumerate(top5_features):
        print(f"  Feature {feat}: used {mean_feature_use[feat]*100:.2f}% of the time")
