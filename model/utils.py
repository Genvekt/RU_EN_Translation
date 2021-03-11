import torch
from torch import nn
import torch.nn.functional as F
import math
import copy

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_seq_len = 200):
        super().__init__()
        self.model_dim = model_dim
        
        pe = torch.zeros(max_seq_len, model_dim, requires_grad=False)
        for pos in range(max_seq_len):
            for i in range(0, model_dim):
                if i % 2 == 0:
                    pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/model_dim)))
                else:
                    pe[pos, i] = math.cos(pos / (10000 ** ((2 * (i + 1))/model_dim)))
                
        self.pe = pe.unsqueeze(0)
 
    
    def forward(self, x):
        S = x.shape[1]
        x = x * math.sqrt(self.model_dim) + self.pe[:,:S]
        return x


class FeedForward(nn.Module):
    def __init__(self, model_dim):
        super().__init__() 
        hidden_dim = model_dim*2

        self.fc1 = nn.Linear(model_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, model_dim)
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    
def clone(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])