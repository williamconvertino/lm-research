import torch.nn as nn
import torch.nn.functional as F
from .model_components.transformer_block import TransformerBlock
from transformers import GPT2Config, GPT2LMHeadModel

class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        config = GPT2Config(
            vocab_size=config.tokenizer.vocab_size,
            n_positions=config.max_seq_len,
            n_embd=config.d_embed,
            n_layer=config.n_layers,
            n_head=config.n_heads,
            dropout=config.dropout,
            bias=False
        )
        self.model = GPT2LMHeadModel(config)

    def forward(self, x, targets=None):
        outputs = self.model(x)
        logits = outputs.logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.contiguous().view(-1), ignore_index=self.tokenizer.pad_id)
        return logits, loss