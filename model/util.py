import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
import numpy as np

class CondTrans(nn.Module):
    def __init__(self, cond_dim):
        super(CondTrans, self).__init__()
        self.cemb1 = nn.Linear(1024, cond_dim)
        self.cemb2 = nn.Linear(1024, cond_dim)
        self.cemb3 = nn.Linear(1024, cond_dim)
        
    def forward(self, cond):
        cond1 = self.cemb1(cond[:, :5, :])
        cond2 = self.cemb2(cond[:, 5:13, :])
        cond3 = self.cemb3(cond[:, 13, :])
        cond_flat = torch.cat([cond1.sum(1), cond2.sum(1), cond3], 1)
        
        return cond, cond_flat

class FiLM_layer(nn.Module):
    def __init__(self, output_dim, cond_totlen, dropout):
        super(FiLM_layer, self).__init__()
        self.condition_layer = nn.Sequential(
            nn.Linear(cond_totlen, cond_totlen),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(cond_totlen, 2*output_dim))
    
    def forward(self, x, cond_flat):
        gamma_beta = self.condition_layer(cond_flat) 
        gamma, beta = gamma_beta.chunk(2, dim=-1)   

        return (gamma.unsqueeze(1)+1) * x + beta.unsqueeze(1)

class AttPool(nn.Module):
    def __init__(self, output_dim=128):
        super(AttPool, self).__init__()
        self.w1 = nn.Linear(output_dim, 1)
        self.sm = nn.Softmax(1)
        self.bm = nn.BatchNorm1d(output_dim)
        
    def forward(self, x_batch):
        x_p = self.bm(torch.sum(self.sm(self.w1(x_batch)) * x_batch, 1))
        return x_p