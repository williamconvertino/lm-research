import torch
import torch.nn as nn
import torch.nn.functional as F

class FMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.fc_1 = nn.Linear(config.embed_dim_f, config.mlp_dim_f, bias=True)
        self.fc_2 = nn.Linear(config.mlp_dim_f, config.embed_dim_f, bias=True)
        
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc_2(x)
        return x
    
class PhiMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.fc_1 = nn.Linear(config.embed_dim_phi, config.mlp_dim_phi, bias=True)
        self.fc_2 = nn.Linear(config.mlp_dim_phi, config.embed_dim_phi, bias=True)
        
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc_2(x)
        return x