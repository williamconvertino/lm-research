import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LMBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.name=self.full_name="LMBase"

    @torch.no_grad()
    def generate(
        self, 
        input_ids, 
        max_generation_length, 
        tokenizer, 
        temperature=1.0, 
        top_p=0.9, 
        return_generation_only=False
    ):

        self.eval()

        batch_size = input_ids.size(0)
        device = input_ids.device

        generated = input_ids.clone()
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_generation_length):

            logits = self(generated)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)

            sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            cutoff_mask = cumulative_probs > top_p
            cutoff_mask[:, 1:] = cutoff_mask[:, :-1].clone()
            cutoff_mask[:, 0] = False

            sorted_probs = sorted_probs.masked_fill(cutoff_mask, 0.0)
            normalized_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

            probs = torch.zeros_like(normalized_probs).scatter(-1, sorted_indices, normalized_probs)

            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            next_token = torch.where(finished, torch.full_like(next_token, tokenizer.pad_token_id), next_token)

            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

            finished |= next_token == tokenizer.eos_token_id

            if finished.all():
                break

        if return_generation_only:
            return generated[:, input_ids.size(1):]
        else:
            return generated

    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        embed_params = sum(p.numel() for name, p in self.named_parameters() if "embed" in name.lower())
        non_embed_params = total_params - embed_params
        return total_params, embed_params, non_embed_params

    def init_weights(self, module, num_layers=None):
        if isinstance(module, nn.Linear):
            std = 0.02 if num_layers is None else 0.02 / math.sqrt(2 * num_layers)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)