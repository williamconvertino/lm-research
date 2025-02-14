import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings
from models.model_util.base_model import BaseModel
from models.model_util.zero_mean_embedding import ZeroMeanEmbedding

class Attention(nn.Module):
    def __init__(self, d_embed, n_heads):
        super().__init__()
        
        self.d_embed = d_embed
        self.n_heads = n_heads
        
        self.W_q = nn.Linear(self.d_embed, self.d_embed, bias=False)
        self.W_k = nn.Linear(self.d_embed, self.d_embed, bias=False)
        self.W_v = nn.Linear(self.d_embed, self.d_embed, bias=False)
        self.out_proj = nn.Linear(self.d_embed, self.d_embed, bias=False)

        self.rotary_embedding = RotaryPositionalEmbeddings(self.d_embed)

        self.attn_dropout = nn.Dropout(0.1)
        self.out_dropout = nn.Dropout(0.1)

    def forward(self, q, k, v):
        B, S, _ = q.shape
        
        q = self.W_q(q) # (B, S, d_attn * n_heads)
        k = self.W_k(k) # (B, S, d_attn * n_heads)
        v = self.W_v(v) # (B, S, d_attn * n_heads)

        q = q.view(B, S, self.n_heads, self.d_embed // self.n_heads) # (B, S, n_heads, d_attn)
        k = k.view(B, S, self.n_heads, self.d_embed // self.n_heads) # (B, S, n_heads, d_attn)
        v = v.view(B, S, self.n_heads, self.d_embed // self.n_heads) # (B, S, n_heads, d_attn)
        
        q = self.rotary_embedding(q)
        k = self.rotary_embedding(k)

        q = q.transpose(1, 2) # (B, n_heads, S, d_attn)
        k = k.transpose(1, 2) # (B, n_heads, S, d_attn)
        v = v.transpose(1, 2) # (B, n_heads, S, d_attn)

        causal_mask = torch.triu(torch.ones(S, S), diagonal=1).bool().logical_not()
        causal_mask = causal_mask.to(q.device)

        attn_scores = k @ q.transpose(-2, -1) / math.sqrt(self.d_embed // self.n_heads) # (B, n_heads, S, S)
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))
        attn_scores = F.softmax(attn_scores, dim=-1)
        attn_scores = self.attn_dropout(attn_scores)

        attn = attn_scores @ v # (B, n_heads, S, d_attn)
        attn = attn.transpose(1, 2).contiguous().view(B, S, self.d_embed) # (B, S, d_embed)
        out = self.out_proj(attn) # (B, S, d_embed)
        out = self.out_dropout(out)

        return out

class FeedForward(nn.Module):
    def __init__(self, d_embed):
        super().__init__()
        self.d_embed = d_embed

        self.ln = nn.LayerNorm(self.d_embed)

        self.fc1 = nn.Linear(self.d_embed, 4 * self.d_embed)
        self.fc2 = nn.Linear(4 * self.d_embed, self.d_embed)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.ln(x)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_embed, n_heads):
        super().__init__()

        self.attention = Attention(d_embed, n_heads)
        self.feed_forward = FeedForward(d_embed)
        
    def forward(self, x):
        x = x + self.attention(x)
        x = x + self.feed_forward(x)
        return x

class GPT(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embedding = ZeroMeanEmbedding(config.vocab_size, config.d_embed)
        
        self.transformer_blocks = nn.ModuleList([TransformerBlock(config.d_embed, config.n_heads) for _ in range(config.n_layers)])

        self.ln_f = nn.LayerNorm(config.d_embed)

        self.lm_head = nn.Linear(config.d_embed, config.vocab_size, bias=False)
        self.embedding.tie_weights(self.lm_head)
        
        self.init_weights()
        
    def forward(self, x, targets=None):
        B, S = x.shape

        x = self.embedding(x) # (B, S, d_embed)

        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.ln_f(x)

        logits = self.lm_head(x)
        if targets is None:
            return logits, None
    
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.contiguous().view(-1))
        return logits, loss