import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from lmr.models.lm_base import LMBase
from .components import ICLBlock, PhiBlock

class FSTClean(LMBase):

    def __init__(self, config, train_mode=False):
        super().__init__()
        
        self.config = config
        self.train_mode = train_mode
        
        self.embedding_f = nn.Embedding(config.vocab_size, config.embed_dim_f)
        
        if not self.config.use_shared_embedding:
            self.embedding_phi = nn.Embedding(config.vocab_size, config.embed_dim_phi) # Allow for a separate lookup table
            
        self.phi_s = nn.Parameter(torch.randn(1, 1, config.embed_dim_phi))
        
        self.phi_blocks = nn.ModuleList([PhiBlock(config) for _ in range(config.n_layers // 2)])
        self.icl_blocks = nn.ModuleList([ICLBlock(config) for _ in range(config.n_layers // 2)])
        
        self.ln_out = nn.LayerNorm(config.embed_dim_f)
        self.lm_head = nn.Linear(config.embed_dim_f, config.vocab_size, bias=False)
        
        self.apply(self.init_weights)
        self.lm_head.weight = self.embedding_f.weight
        
    def forward(self, input_ids):
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        context_embeddings = self.embedding_f(input_ids)
        
        if self.config.use_shared_embedding:
            phi = context_embeddings
            if self.config.embed_dim_f != self.config.embed_dim_phi:
                phi = phi[:, :, :self.config.embed_dim_phi]
        else:
            phi = self.embedding_phi(input_ids)
            
        context_embeddings = torch.cat([context_embeddings, torch.zeros(batch_size, 1, self.config.embed_dim_f, device=device)], dim=1) # Add zero vector to the end (for prediction N+1)
        
        phi_s = self.phi_s.expand(batch_size, -1, -1)
        phi = torch.cat([phi_s, phi], dim=1) # Append starting token
        
        if self.config.use_f_resid:
            f = torch.zeros(batch_size, seq_len + 1, self.config.embed_dim_f, device=device) # Initialize as zero
            f_mlp = torch.zeros(batch_size, seq_len + 1, self.config.embed_dim_f, device=device) # Initialize as zero
            for phi_block, icl_block in zip(self.phi_blocks, self.icl_blocks):
                phi = phi_block(phi)
                _, _, f, block_f_mlp = icl_block(phi, context_embeddings, f, f_mlp)
                f_mlp += block_f_mlp
        else:
            f_mlp = torch.zeros(batch_size, seq_len + 1, self.config.embed_dim_f, device=device) # Initialize as zero
            for phi_block, icl_block in zip(self.phi_blocks, self.icl_blocks):
                phi = phi_block(phi)
                f_mlp += icl_block(phi, context_embeddings, f, f_mlp)
                
        f_NP1 = f_mlp[:, 1:, :]

        f_NP1 = self.ln_out(f_NP1)
        logits = self.lm_head(f_NP1)
        
        return logits

    def calculate_loss(self, logits, target_tokens, l1_loss_lambda=None):    
        
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_tokens.reshape(-1),
            reduction='mean'
            )
        
        if l1_loss_lambda is not None:
            # Add penalty on ICL MLPs to induce sparsity
            for icl_block in self.icl_blocks:
                l1 = self.mlp.weight.abs().sum()
                loss += l1 * l1_loss_lambda
        
        return loss