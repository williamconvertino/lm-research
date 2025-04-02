import random
import torch
import torch.nn.functional as F
from .trainer import Trainer

def generate_text_greedy(model, tokenizer, prompt, max_length=50, temperature=1.0, device="gpu"):
    model.to(device)
    model.eval()
    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    generated = input_ids
    input_size = input_ids.size(1)
    
    for _ in range(max_length):
        if generated.size(1) > model.config.max_seq_len:
            input_ids = generated[:, -model.config.max_seq_len:]
        else:
            input_ids = generated

        logits, _ = model(input_ids)
        next_token_logits = logits[:, -1, :] / temperature
        probabilities = F.softmax(next_token_logits, dim=-1)
        
        next_token = torch.multinomial(probabilities, num_samples=1)
        generated = torch.cat((generated, next_token), dim=1).to(device)
        
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    return tokenizer.decode(generated[0].tolist()[input_size:])

def generate_text_beam(model, tokenizer, prompt, max_length=50, beam_width=3, device="gpu"):
    model.to(device)
    model.eval()
    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    input_size = input_ids.size(1)
    beams = [(input_ids, 0)] 
    
    for _ in range(max_length):
        new_beams = []
        
        for seq, score in beams:
            if seq.size(1) > model.config.max_seq_len:
                seq_input = seq[:, -model.config.max_seq_len:]
            else:
                seq_input = seq
            
            logits, _ = model(seq_input)
            next_token_logits = logits[:, -1, :]
            log_probs = F.log_softmax(next_token_logits, dim=-1)
            
            topk_log_probs, topk_indices = torch.topk(log_probs, beam_width, dim=-1)
            
            for i in range(beam_width):
                next_token = topk_indices[0, i].unsqueeze(0).unsqueeze(0)
                new_seq = torch.cat([seq, next_token], dim=1).to(device)
                new_score = score + topk_log_probs[0, i].item()
                new_beams.append((new_seq, new_score))
        
        new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        beams = new_beams
        
        if all(seq[0, -1].item() == tokenizer.eos_token_id for seq, _ in beams):
            break

    best_seq = beams[0][0]
    return tokenizer.decode(best_seq[0].tolist()[input_size:])

def generate_text_topk(model, tokenizer, prompt, max_length=50, temperature=1.0, top_k=40, device="gpu"):
    """
    Generate text using top-k sampling.
    """
    model.to(device)
    model.eval()
    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    generated = input_ids
    input_size = input_ids.size(1)
    
    for _ in range(max_length):
        if generated.size(1) > model.config.max_seq_len:
            input_ids = generated[:, -model.config.max_seq_len:]
        else:
            input_ids = generated

        logits, _ = model(input_ids)
        next_token_logits = logits[:, -1, :] / temperature
        # Keep only the top_k tokens
        topk_values, topk_indices = torch.topk(next_token_logits, top_k, dim=-1)
        filtered_logits = torch.full_like(next_token_logits, float('-inf'))
        filtered_logits.scatter_(dim=-1, index=topk_indices, src=next_token_logits.gather(dim=-1, index=topk_indices))
        probabilities = F.softmax(filtered_logits, dim=-1)
        
        next_token = torch.multinomial(probabilities, num_samples=1)
        generated = torch.cat((generated, next_token), dim=1).to(device)
        
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    return tokenizer.decode(generated[0].tolist()[input_size:])

def generate_text_nucleus(model, tokenizer, prompt, max_length=50, temperature=1.0, top_p=0.9, device="gpu"):
    """
    Generate text using nucleus (top-p) sampling.
    """
    model.to(device)
    model.eval()
    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    generated = input_ids
    input_size = input_ids.size(1)
    
    for _ in range(max_length):
        if generated.size(1) > model.config.max_seq_len:
            input_ids = generated[:, -model.config.max_seq_len:]
        else:
            input_ids = generated
            
        logits, _ = model(input_ids)
        next_token_logits = logits[:, -1, :] / temperature
        
        # Compute probabilities and sort them
        probs = F.softmax(next_token_logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to ensure at least one token is kept
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = torch.zeros_like(next_token_logits, dtype=torch.bool)
        indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        next_token_logits = next_token_logits.masked_fill(indices_to_remove, float('-inf'))
        
        probabilities = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        generated = torch.cat((generated, next_token), dim=1).to(device)
        
        if next_token.item() == tokenizer.eos_token_id:
            break
            
    return tokenizer.decode(generated[0].tolist()[input_size:])

class Evaluator:
    def __init__(self, model, splits, tokenizer):
        self.model = model
        self.splits = splits
        self.tokenizer = tokenizer

        self.device = self._get_device()

    def _get_device(self):
        if not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            return torch.device('cpu')
        vram_required = 5
        print(f"Estimated VRAM required: {vram_required:.2f}GB")
        for i in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(i)
                gpu = torch.device(f'cuda:{i}')
                free_memory, total_memory = torch.cuda.mem_get_info(gpu)
                total_memory = int(total_memory / 1024**3)
                free_memory = int(free_memory / 1024**3)  
                if free_memory > vram_required:
                    print(f"Using GPU [{i}]: {props.name} with {free_memory:.2f}GB")
                    return torch.device(f'cuda:{i}')
                else:
                    print(f"GPU [{i}]: {props.name} only has {free_memory:.2f}GB free memory, skipping")
            except Exception:
                print(f"Error reading GPU [{i}], skipping")
        raise RuntimeError(f"No GPU with at least {vram_required}GB of free memory found")
    
    def eval(self, eval_types, num_prompts=10, temperature=1.0, top_k=40, top_p=0.9, beam_width=3):
        """
        Evaluate multiple sampling strategies (e.g. "greedy", "beam", "topk", "nucleus")
        back-to-back for each prompt in the test split.
        """
        prompts = []
        
        # Gather the first `num_prompts` prompts from the test split.
        for i, batch in enumerate(self.splits["test"]):
            if i >= num_prompts:
                break
            example = batch[0]
            prompt_tokens = example.tolist()[:50]  # Use first 50 tokens as prompt
            prompt_tokens = [token for token in prompt_tokens if token != self.tokenizer.pad_token_id]
            prompt_text = self.tokenizer.decode(prompt_tokens)
            prompts.append(prompt_text)
        
        # Mapping of evaluation type to corresponding generation function.
        eval_funcs = {
            "greedy": lambda prompt: generate_text_greedy(self.model, self.tokenizer, prompt, temperature=temperature, device=self.device),
            "beam": lambda prompt: generate_text_beam(self.model, self.tokenizer, prompt, beam_width=beam_width, device=self.device),
            "topk": lambda prompt: generate_text_topk(self.model, self.tokenizer, prompt, temperature=temperature, top_k=top_k, device=self.device),
            "nucleus": lambda prompt: generate_text_nucleus(self.model, self.tokenizer, prompt, temperature=temperature, top_p=top_p, device=self.device),
        }
        
        # For each prompt, evaluate each of the requested sampling methods.
        for prompt in prompts:
            print("Prompt:")
            print(prompt)
            print("-" * 50)
            for eval_type in eval_types:
                if eval_type in eval_funcs:
                    print(f"{eval_type.capitalize()} Sampling Generation:")
                    output = eval_funcs[eval_type](prompt)
                    print(output)
                    print("-" * 50)
                else:
                    print(f"Unknown evaluation type: {eval_type}")
                    print("-" * 50)
            print("*" * 50)
                    
    def eval_loss(self):
        trainer = Trainer(self.model, self.splits, self.tokenizer, self.device)
        loss = trainer.validate()
        print("="*50)
        print(f"Validation Loss: {loss:.4f}")
        print("="*50)