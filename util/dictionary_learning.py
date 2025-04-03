import torch
import os
from .sae import SparseAutoencoder, train_sae

class DictionaryLearning:
    
    def __init__(self, model, splits, k=1):
        self.model = model
        self.splits = splits
        self.k = k
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_path = f"data/dictionary_learning/{model.config.name}"
        
    def collect_data(self):
        
        if os.path.exists(self.save_path):
            print(f"Data already collected. Loading from {self.save_path}")
            return
        
        print("Collecting neuron data...")

        os.makedirs(self.save_path, exist_ok=True)
        
        self.model.eval()
        self.model.to(self.device)
        self.model.config.gather_neurons = True
        
        with torch.no_grad():
            for i, batch in enumerate(self.splits["test"]):
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

                torch.save(neurons, os.path.join(self.save_path, f"neurons_{i}.pt"))
                if i % 100 == 0:
                    print(f"\rCollected neuron data for batch {i}", end="")

        self.model.config.gather_neurons = False

    def run_dictionary_learning(self):
        
        train_sae(self.model.config, k=self.k, layer=0, sublayer='ff')