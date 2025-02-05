import time
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
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )
        
        warmup_steps = 1000  # Fixed warmup steps, since we don't have set epochs
        self.scheduler = self.get_scheduler(warmup_steps)
        
        # Early stopping parameters
        self.patience = config.get('patience', 5)
        self.early_stopping_counter = 0
        
        self.grad_clip = config.get('grad_clip', 1.0)

    def get_device(self):
        if not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            return torch.device('cpu')
        print("Using GPU")
        return torch.device('cuda')

    def get_scheduler(self, warmup_steps):
        from transformers import get_scheduler
        estimated_total_steps = 100000
        return get_scheduler(
            "cosine_with_warmup",
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=estimated_total_steps
        )

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
                self.scheduler.step()

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
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, "best_model.pt")
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