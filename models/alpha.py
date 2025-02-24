import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.W_q = nn.Linear(config.d_embed, config.d_embed, bias=False)
        self.W_k = nn.Linear(config.d_embed, config.d_embed, bias=False)
        self.W_v = nn.Linear(config.d_embed, config.d_embed, bias=False)
        self.W_o = nn.Linear(config.d_embed, config.d_embed, bias=False)
        
        self.attn_scale = 1 / math.sqrt(config.d_embed)
        
        self.rotary_embeddings = RotaryPositionalEmbeddings(config.d_embed // config.n_heads)
        
        self.drop_attn = nn.Dropout(0.1)
        self.drop_resid = nn.Dropout(0.1)
        
    def forward(self, q, k=None, v=None):
        
        B, S, E = q.shape
        
        if k is None:
            k = q
        if v is None:
            v = q
            
        q = self.W_q(q) # (B, S, d_embed)
        k = self.W_k(k)
        v = self.W_v(v)
        
        q = q.view(B, S, self.config.n_heads, self.config.d_embed // self.config.n_heads) # (B, S, n_heads, d_embed // n_heads)
        k = k.view(B, S, self.config.n_heads, self.config.d_embed // self.config.n_heads) # (B, S, n_heads, d_embed // n_heads)
        v = v.view(B, S, self.config.n_heads, self.config.d_embed // self.config.n_heads) # (B, S, n_heads, d_embed // n_heads)
        
        q = self.rotary_embeddings(q)
        k = self.rotary_embeddings(k)
        
        q = q.transpose(1, 2) # (B, n_heads, S, d_embed // n_heads)
        k = k.transpose(1, 2) # (B, n_heads, S, d_embed // n_heads)
        v = v.transpose(1, 2) # (B, n_heads, S, d_embed // n_heads)
        
        causal_mask = torch.triu(torch.ones(S, S), diagonal=1).bool().to(q.device)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.attn_scale
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.drop_attn(attn_probs)
        
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.config.d_embed)
        attn_output = self.W_o(attn_output)
        attn_output = self.drop_resid(attn_output)
        
        return attn_output
        
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.fc_1 = nn.Linear(config.d_embed, 4 * config.d_embed)
        self.fc_2 = nn.Linear(4 * config.d_embed, config.d_embed)
        
        self.activation = nn.GELU()    
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.fc_2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.use_ff = getattr(config, 'use_ff', True)
        
        print(f"use_ff={self.use_ff}")
        
        self.ln_1 = nn.LayerNorm(config.d_embed)
        self.attention = Attention(config)
        
        if self.use_ff:
            self.ln_2 = nn.LayerNorm(config.d_embed)
            self.feed_forward = FeedForward(config)
        
    def forward_inference(self, x, ff_out):
        
        ff_in = ff_out + x
        x = x + self.attention(self.ln_1(ff_in))
        ff_out = self.feed_forward(self.ln_2(x))
        
        return x, ff_out

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        if self.use_ff:
            x = x + self.feed_forward(self.ln_2(x))
        return x

class Alpha(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_embed)

        self.transformer_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        
        self.ln_f = nn.LayerNorm(config.d_embed)

        self.lm_head = nn.Linear(config.d_embed, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, x, targets=None, ignore_index=-1):
        
        B, S = x.shape

        x = self.embedding(x)
        
        if targets is not None:
            for block in self.transformer_blocks:
                x = block(x)
        else:
            ff_out = torch.zeros_like(x)
            for i, block in enumerate(self.transformer_blocks):
                if i == len(self.transformer_blocks) - 1:
                    x = x + ff_out
                    x = block(x)
                else:
                    x, ff_out = block.forward_inference(x, ff_out)
                
        x = self.ln_f(x)
        
        logits = self.lm_head(x)
        
        if targets is None:
            return logits, None
    
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.contiguous().view(-1), ignore_index=ignore_index)
        return logits, loss