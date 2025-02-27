import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.W_q = nn.Linear(2 * config.d_tri, config.d_embed, bias=False)
        self.W_k = nn.Linear(2 * config.d_tri, config.d_embed, bias=False)
        self.W_v = nn.Linear(config.d_tri, config.d_tri, bias=False)
        self.W_o = nn.Linear(config.d_tri, config.d_tri, bias=False)
        
        self.attn_scale = 1 / math.sqrt(config.d_embed)
        
        self.rotary_embeddings = RotaryPositionalEmbeddings(config.d_embed // config.n_heads)
        
        self.drop_attn = nn.Dropout(0.1)
        self.drop_resid = nn.Dropout(0.1)
        
    def forward(self, q, k, v):
        
        B, S, E = q.shape
            
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

        self.fc_1 = nn.Linear(2 * config.d_tri, 4 * config.d_embed)
        self.fc_2 = nn.Linear(4 * config.d_embed, config.d_tri)
        
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
        
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ln_1 = nn.LayerNorm(2 * config.d_tri)
        self.ln_2 = nn.LayerNorm(2 * config.d_tri)
        
    def forward(self, ex, f_plus):
        
        q = k = self.ln_1(f_plus)
        v = ex
        
        f_plus = f_plus + self.attention(q=q, k=k, v=v)        
        
        ex = ex + self.feed_forward(self.ln_2(f_plus))
        
        return ex, f_plus

class B0(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        config.d_triple = config.d_embed // 3
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_triple)

        self.transformer_blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layers)])
        
        self.ff_out = FeedForward(config)
        self.ln_f = nn.LayerNorm(config.d_triple)

        self.lm_head = nn.Linear(config.d_triple, config.vocab_size, bias=False)
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
        
        with torch.no_grad():
            self.embedding.weight -= self.embedding.weight.mean(0, keepdim=True) # Zero mean

        ex = self.embedding(x) # Start with E[W_c] = 0
        
        f_plus = torch.cat([ex, torch.zeros_like(ex)], dim=-1) # Combined f and covariate terms
        
        ex, f_plus = self.transformer_blocks(ex, f_plus)
        
        f = self.ff_out(f_plus)
        
        f = self.ln_f(f)
        
        logits = self.lm_head(f)
        
        if targets is None:
            return logits, None
    
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.contiguous().view(-1), ignore_index=ignore_index)
        return logits, loss