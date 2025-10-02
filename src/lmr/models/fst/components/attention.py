import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings
from torch.nn.attention import flex_attention, create_block_mask

class ICLAttention(nn.Module):
    def __init__(self, config, use_wv_icl=True):
        super().__init__()
        
        self.config = config
        
        self.W_q = nn.Linear(config.embed_dim_phi, config.hidden_dim_f, bias=False) # We set biases according to https://arxiv.org/pdf/2302.08626
        self.W_k = nn.Linear(config.embed_dim_phi, config.hidden_dim_f, bias=False)
    
        self.W_v = nn.Linear(config.embed_dim_f, config.hidden_dim_f, bias=True)
        self.W_o = nn.Linear(config.hidden_dim_f, config.embed_dim_f, bias=True)
        
        self.attn_scale = 1 / math.sqrt(config.hidden_dim_f // config.n_heads_f)
        
        self.drop_attn = nn.Dropout(0.1)
        self.drop_resid = nn.Dropout(0.1)
        
        # Need to modify the causal mask to properly manifest a GD update
        causal_mask = torch.triu(torch.ones(config.max_seq_len + 10, config.max_seq_len + 10), diagonal=0).bool()
        causal_mask[0, 0] = False
        
        self.register_buffer("causal_mask", causal_mask)
        
    def forward(self, q, k, v):
        
        B, S, E = q.shape
        
        q = self.W_q(q).view(B, S, self.config.n_heads_f, self.config.hidden_dim_f // self.config.n_heads_f).transpose(1, 2)
        k = self.W_k(k).view(B, S, self.config.n_heads_f, self.config.hidden_dim_f // self.config.n_heads_f).transpose(1, 2)
        
        v = self.W_v(v).view(B, S, self.config.n_heads_f, self.config.hidden_dim_f // self.config.n_heads_f).transpose(1, 2)
        
        # Note the lack of rotary embeddings in ICL attention
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.attn_scale
        attn_scores = attn_scores.masked_fill(self.causal_mask[:S, :S], float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.drop_attn(attn_probs)
        
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        attn_output = attn_output.view(B, S, self.config.hidden_dim_f)
            
        attn_output = self.W_o(attn_output)
        attn_output = self.drop_resid(attn_output)
        
        return attn_output
    
# Vanilla Attention, but with dimensions properly aligned
class PhiAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.W_q = nn.Linear(config.embed_dim_phi, config.hidden_dim_phi, bias=False)
        self.W_k = nn.Linear(config.embed_dim_phi, config.hidden_dim_phi, bias=False)
        self.W_v = nn.Linear(config.embed_dim_phi, config.hidden_dim_phi, bias=True)
        self.W_o = nn.Linear(config.hidden_dim_phi, config.embed_dim_phi, bias=True)
        
        self.attn_scale = 1 / math.sqrt(config.hidden_dim_phi // config.n_heads_phi)
        
        self.rotary_embeddings = RotaryPositionalEmbeddings(config.hidden_dim_phi // config.n_heads_phi, max_seq_len=config.max_seq_len + 10)
        
        self.drop_attn = nn.Dropout(0.1)
        self.drop_resid = nn.Dropout(0.1)
        
        causal_mask = torch.triu(torch.ones(config.max_seq_len + 10, config.max_seq_len + 10), diagonal=1).bool()
        
        self.register_buffer("causal_mask", causal_mask)
    
    def forward(self, x):
        
        B, S, E = x.shape
        
        q = self.W_q(x).view(B, S, self.config.n_heads_phi, self.config.hidden_dim_phi // self.config.n_heads_phi).transpose(1, 2)
        k = self.W_k(x).view(B, S, self.config.n_heads_phi, self.config.hidden_dim_phi // self.config.n_heads_phi).transpose(1, 2)
        v = self.W_v(x).view(B, S, self.config.n_heads_phi, self.config.hidden_dim_phi // self.config.n_heads_phi).transpose(1, 2)
        
        q = self.rotary_embeddings(q)
        k = self.rotary_embeddings(k)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.attn_scale
        attn_scores = attn_scores.masked_fill(self.causal_mask[:S, :S], float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.drop_attn(attn_probs)
        
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.config.hidden_dim_phi)
        attn_output = self.W_o(attn_output)
        attn_output = self.drop_resid(attn_output)
        
        return attn_output