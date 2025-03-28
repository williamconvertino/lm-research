import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings


class GAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.W_q = nn.Linear(config.d_embed // 2, config.n_heads * config.d_embed, bias=False)
        self.W_k = nn.Linear(config.d_embed // 2, config.n_heads * config.d_embed, bias=False)
        self.W_v = nn.Linear(config.d_embed // 2, config.n_heads * config.d_embed, bias=False)
        self.W_o = nn.Linear(config.n_heads * config.d_embed, config.d_embed // 2, bias=False)
        
        self.attn_scale = 1 / math.sqrt(config.d_embed)
        
        self.rotary_embeddings = RotaryPositionalEmbeddings(config.d_embed)
        
        self.drop_attn = nn.Dropout(0.1)
        self.drop_resid = nn.Dropout(0.1)
        
    def forward(self, q, k=None, v=None):
        
        B, S, E = q.shape
        
        if k is None:
            k = q
        if v is None:
            v = q
        
        q = self.W_q(q).view(B, S, self.config.n_heads, self.config.d_embed).transpose(1, 2)
        k = self.W_k(k).view(B, S, self.config.n_heads, self.config.d_embed).transpose(1, 2)
        v = self.W_v(v).view(B, S, self.config.n_heads, self.config.d_embed).transpose(1, 2)
        
        q = self.rotary_embeddings(q)
        k = self.rotary_embeddings(k)
        
        causal_mask = torch.triu(torch.ones(S, S), diagonal=1).bool().to(q.device)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.attn_scale
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.drop_attn(attn_probs)
        
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.config.d_embed * self.config.n_heads)
        attn_output = self.W_o(attn_output)
        attn_output = self.drop_resid(attn_output)
        
        return attn_output

class GFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.fc_1 = nn.Linear(config.d_embed, 4 * config.d_embed)
        self.fc_2 = nn.Linear(4 * config.d_embed, config.d_embed // 2)
        
        self.activation = nn.GELU()    
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.fc_2(x)
        return x

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.W_q = nn.Linear(config.d_embed, config.n_heads * config.d_embed, bias=False)
        self.W_k = nn.Linear(config.d_embed, config.n_heads * config.d_embed, bias=False)
        self.W_v = nn.Linear(config.d_embed, config.n_heads * config.d_embed, bias=False)
        self.W_o = nn.Linear(config.n_heads * config.d_embed, config.d_embed, bias=False)
        
        self.attn_scale = 1 / math.sqrt(config.d_embed)
        
        self.rotary_embeddings = RotaryPositionalEmbeddings(config.d_embed)
        
        self.drop_attn = nn.Dropout(0.1)
        self.drop_resid = nn.Dropout(0.1)
        
    def forward(self, q, k=None, v=None):
        
        B, S, E = q.shape
        
        if k is None:
            k = q
        if v is None:
            v = q
        
        q = self.W_q(q).view(B, S, self.config.n_heads, self.config.d_embed).transpose(1, 2)
        k = self.W_k(k).view(B, S, self.config.n_heads, self.config.d_embed).transpose(1, 2)
        v = self.W_v(v).view(B, S, self.config.n_heads, self.config.d_embed).transpose(1, 2)
        
        q = self.rotary_embeddings(q)
        k = self.rotary_embeddings(k)
        
        causal_mask = torch.triu(torch.ones(S, S), diagonal=1).bool().to(q.device)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.attn_scale
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.drop_attn(attn_probs)
        
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.config.d_embed * self.config.n_heads)
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
        
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ln_1 = nn.LayerNorm(config.d_embed)
        self.ln_2 = nn.LayerNorm(config.d_embed)
        
    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.feed_forward(self.ln_2(x))
        return x

class GBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.attention = GAttention(config)
        self.feed_forward = GFeedForward(config)
        
        self.ln_f = nn.LayerNorm(config.d_embed)
        self.ln_g = nn.LayerNorm(config.d_embed)
        self.ln_2 = nn.LayerNorm(config.d_embed)
        
    def forward(self, f, g):
        
        qk = self.ln_f(f)
        v = self.ln_g(g)

        f = f + self.attention(q=qk, k=qk, v=v)
        
        x = torch.cat([f, g], dim=-1)
        
        g = g + self.feed_forward(self.ln_2(x))
        return f, g

class DivFormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_embed // 2)

        self.g_blocks = nn.ModuleList([GBlock(config) for _ in range(config.n_layers - 1)])
        self.transformer_block = TransformerBlock(config)
        
        self.ln_f = nn.LayerNorm(config.d_embed //2)

        self.lm_head = nn.Linear(config.d_embed // 2, config.vocab_size, bias=False)
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

        f = g = self.embedding(x)
        
        for g_block in self.g_blocks:
            f, g = g_block(f, g)
        
        x = torch.cat([f, g], dim=-1)
        x = self.transformer_block(x)
        
        x = x[:, :, :self.config.d_embed // 2]
        
        x = self.ln_f(x)
        
        logits = self.lm_head(x)
        
        if targets is None:
            return logits, None
    
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.contiguous().view(-1), ignore_index=ignore_index)
        return logits, loss