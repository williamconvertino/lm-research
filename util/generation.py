import torch
import torch.nn.functional as F

def generate_nucleus(model, tokenizer, input_ids, max_length=50, temperature=1.0, top_p=0.9, device=None):
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    len_input_ids = len(input_ids)
    
    model.to(device)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
    
    generated = input_ids
    
    for _ in range(max_length):
        if generated.size(1) > model.config.max_seq_len:
            input_ids = generated[:, -model.config.max_seq_len:]
        else:
            input_ids = generated
            
        logits, _ = model(input_ids)
        next_token_logits = logits[:, -1, :] / temperature
        
        probs = F.softmax(next_token_logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = torch.zeros_like(next_token_logits, dtype=torch.bool)
        indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        next_token_logits = next_token_logits.masked_fill(indices_to_remove, float('-inf'))
        
        probabilities = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        
        if next_token.item() == tokenizer.eos_token_id:
            break
        
        generated = torch.cat((generated, next_token), dim=1).to(device)
         
    return generated[0].tolist()[len_input_ids:]