import random
import torch
import torch.nn.functional as F

def generate_text_greedy(model, tokenizer, prompt, max_length=50, temperature=1.0, device="gpu"):
    model.to(device)
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
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
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
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

class Evaluator:
    def __init__(self, model, splits, tokenizer):
        self.model = model
        self.test_loader = splits["test"]
        self.tokenizer = tokenizer

        self.device = self.get_device()

    def get_device(self):
        if not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            return torch.device('cpu')
        print("Using GPU")
        return torch.device('cuda')
    
    def eval_greedy(self, num_prompts=10):
        prompts = []
        
        for i, batch in enumerate(self.test_loader):
            if i >= num_prompts:
                break
            example = batch[0]
            prompt_text = self.tokenizer.decode(example.tolist()[:50])  # Use first 50 tokens as prompt
            prompts.append(prompt_text)
        
        for prompt in prompts:
            print("Prompt:")
            print(prompt)
            print("-"*50)
            print("Generation:")
            print(generate_text_greedy(self.model, self.tokenizer, prompt, device=self.device))
            print("*"*50)

    def eval_beam(self, num_prompts=10):
        prompts = []
            
        for i, batch in enumerate(self.test_loader):
            if i >= num_prompts:
                break
            example = batch[0]
            prompt_text = self.tokenizer.decode(example.tolist()[:50])  # Use first 50 tokens as prompt
            prompts.append(prompt_text)

        for prompt in prompts:
            print("Prompt:")
            print(prompt)
            print("-"*50)
            print("Beam Output:")
            print(generate_text_beam(self.model, self.tokenizer, prompt, device=self.device))
            print("*"*50)