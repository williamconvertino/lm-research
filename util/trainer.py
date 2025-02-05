import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import math

class Trainer:
    def __init__(self, model, config, train_loader, val_loader):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = self.get_device()

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Early stopping parameters
        self.patience = config.patience
        self.early_stopping_counter = 0
        
        self.grad_clip = config.grad_clip

    def get_device(self):
        if not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            return torch.device('cpu')
        # Choose first GPU with at least 10GB of free memory 
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            if props.total_memory > 10e9:
                print(f"Using GPU [{i}]: {props.name} with {props.total_memory/1e9:.2f}GB")
                return torch.device(f'cuda:{i}')
        raise RuntimeError("No GPU with at least 10GB of free memory found")
    
    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(self.device)
                if input_ids.size(1) < 2:
                    continue
                x = input_ids[:, :-1]
                targets = input_ids[:, 1:]
                _, loss = self.model(x, targets=targets)
                total_loss += loss.item()
                total_batches += 1
        return total_loss / total_batches if total_batches > 0 else 0.0

    def train(self):
        self.model.to(self.device)
        self.model.train()
        start_time = time.time()
        best_val_loss = float("inf")
        epoch = 1

        while True:
            epoch_loss = 0.0
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
            
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(self.device)
                if input_ids.size(1) < 2:
                    continue  
                x = input_ids[:, :-1]
                targets = input_ids[:, 1:]

                self.optimizer.zero_grad()
                _, loss = self.model(x, targets=targets)
                loss.backward()
                
                # Add gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                self.optimizer.step()

                epoch_loss += loss.item()
                
                # Calculate and display perplexity
                perplexity = math.exp(loss.item())
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'ppl': f"{perplexity:.2f}"
                })

            avg_train_loss = epoch_loss / len(self.train_loader)
            val_loss = self.evaluate(self.val_loader)
            val_perplexity = math.exp(val_loss)
            
            print(f"\nEpoch {epoch} complete. Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Perplexity: {val_perplexity:.2f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if not os.path.exists("checkpoints"):
                    os.mkdir("checkpoints")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, f"checkpoints/{self.model.config.name}.pt")
                print(f"Saved new best model with Val Loss: {best_val_loss:.4f}")
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= self.patience:
                    print(f"Early stopping triggered after {epoch} epochs")
                    break
            
            epoch += 1

        total_time = time.time() - start_time
        
        print(f"Training complete in {total_time/60:.2f} minutes.")