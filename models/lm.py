import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings
from models.model_util.base_model import BaseModel

class Attention(nn.Module):
    def __init__(self, d_embed, n_heads):
        super().__init__()
        
        self.d_embed = d_embed
        self.n_heads = n_heads
        
        self.W_q = nn.Parameter(torch.zeros(n_heads, d_embed, d_embed // n_heads))
        self.W_k = nn.Parameter(torch.zeros(n_heads, d_embed, d_embed // n_heads))
        self.W_v = nn.Parameter(torch.zeros(n_heads, d_embed, d_embed // n_heads))
        self.W_o = nn.Linear(d_embed, d_embed, bias=False)

        self.rotary_embedding = RotaryPositionalEmbeddings(self.d_embed // self.n_heads)

        self.attn_dropout = nn.Dropout(0.1)
        self.out_dropout = nn.Dropout(0.1)

    def forward(self, q, k, v):
        B, S, _ = q.shape
        
        q = q.repeat(1, 1, self.n_heads).view(B, S, self.n_heads, self.d_embed).transpose(1, 2)
        k = k.repeat(1, 1, self.n_heads).view(B, S, self.n_heads, self.d_embed).transpose(1, 2)
        v = v.repeat(1, 1, self.n_heads).view(B, S, self.n_heads, self.d_embed).transpose(1, 2)
        
        Q = q @ self.W_q
        K = k @ self.W_k
        V = v @ self.W_v
        
        causal_mask = torch.tril(torch.ones(S, S, device=q.device), diagonal=0).view(1, S, S).bool().logical_not()

        attn_scores = torch.matmul(Q, K.transpose(-1, -2))
        attn_scores = attn_scores / math.sqrt(self.d_embed)
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        attn_scores = F.softmax(attn_scores, dim=-1)
        attn_scores = self.attn_dropout(attn_scores)
        
        attn_output = attn_scores @ V

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.d_embed)
        
        attn_output = self.W_o(attn_output)
        attn_output = self.out_dropout(attn_output)
        
        return attn_output

class FeedForward(nn.Module):
    def __init__(self, d_embed):
        super().__init__()

        self.d_embed = d_embed

        self.fc1 = nn.Linear(self.d_embed, 4 * self.d_embed)
        self.fc2 = nn.Linear(4 * self.d_embed, self.d_embed)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_embed, n_heads):
        super().__init__()

        self.ln_1 = nn.LayerNorm(d_embed)
        self.ln_2 = nn.LayerNorm(d_embed)
        
        self.attention = Attention(d_embed, n_heads)
        self.feed_forward = FeedForward(d_embed)
        
    def forward(self, x):
        x = self.ln_1(x)
        x = x + self.attention(x, x, x)
        x = self.ln_2(x)
        x = x + self.feed_forward(x)
        return x

class LM(BaseModel):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        d_embed = config.d_embed
        n_heads = config.n_heads

        self.embedding = nn.Embedding(config.vocab_size, d_embed)
        
        self.transformer_blocks = nn.ModuleList([TransformerBlock(d_embed, n_heads) for _ in range(config.n_layers)])

        self.ln_f = nn.LayerNorm(d_embed)

        self.lm_head = nn.Linear(d_embed, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        
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