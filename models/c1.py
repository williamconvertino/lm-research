import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.W_q = nn.Parameter(torch.zeros(config.n_heads, config.d_embed, config.d_embed))
        self.W_k = nn.Parameter(torch.zeros(config.n_heads, config.d_embed, config.d_embed))
        # self.W_v = nn.Parameter(torch.zeros(config.n_heads, config.d_tri, config.d_tri))
        self.W_v_diag = nn.Parameter(torch.zeros(config.n_heads, config.d_tri))
        
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
        
        q = q.unsqueeze(2).expand(B, S, self.config.n_heads, 2 * self.config.d_tri) # (B, S, n_heads, d_embed)
        k = k.unsqueeze(2).expand(B, S, self.config.n_heads, 2 * self.config.d_tri)
        v = v.unsqueeze(2).expand(B, S, self.config.n_heads, self.config.d_tri)
            
        q = torch.einsum('b s h e, h e d -> b s h d', q, self.W_q) # (B, S, n_heads, d_embed)
        k = torch.einsum('b s h e, h e d -> b s h d', k, self.W_k)
        W_v = torch.diag_embed(self.W_v_diag)  # (n_heads, d_tri, d_tri)
        v = torch.einsum('b s h e, h e d -> b s h d', v, W_v)
        
        q = self.rotary_embeddings(q) # (B, S, n_heads, d_embed)
        k = self.rotary_embeddings(k)
        
        q = q.transpose(1, 2) # (B, n_heads, S, d_embed)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        causal_mask = torch.triu(torch.ones(S, S), diagonal=1).bool().to(q.device)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.attn_scale
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.drop_attn(attn_probs)
        
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2) # (B, S, n_heads, d_embed)
        
        attn_output = torch.sum(attn_output, dim=2) # (B, S, d_embed)
        
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
        
    def forward(self, e, ex, f):
        
        q = k = torch.cat([e, ex, f], dim=-1)
        v = ex
        
        f = f + self.attention(q=q, k=k, v=v)
        
        ex = e + self.feed_forward(self.ln_2(torch.cat([e, f], dim=-1)))
        
        return e, ex, f

class C1(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        config.d_tri = config.d_embed // 3
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_tri)

        self.transformer_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        
        self.ff_out = FeedForward(config)
        self.ln_f = nn.LayerNorm(config.d_tri)

        self.lm_head = nn.Linear(config.d_tri, config.vocab_size, bias=False)
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

        e = self.embedding(x) # Start with E[W_c] = 0
        ex = e
        f = torch.zeros_like(ex)
        
        for block in self.transformer_blocks:
            e, ex, f = block(e, ex, f)
        
        f = self.ff_out(torch.cat([e, f], dim=-1))
        
        f = self.ln_f(f)
        
        logits = self.lm_head(f)
        
        if targets is None:
            return logits, None
    
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.contiguous().view(-1), ignore_index=ignore_index)
        return logits, loss