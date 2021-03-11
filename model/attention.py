import torch
from torch import nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, heads_num, model_dim):
        super().__init__()
        
        self.h_num = heads_num
        self.model_dim = model_dim
        self.h_dim = model_dim // heads_num
        
        self.q_fc = nn.Linear(model_dim, model_dim)
        self.v_fc = nn.Linear(model_dim, model_dim)
        self.k_fc = nn.Linear(model_dim, model_dim)
        
        self.dropout = nn.Dropout(0.1)
        self.out_fc = nn.Linear(model_dim, model_dim)
        
    def head_attention(self, q, k, v, mask=None):
        # [B, h_num, S, h_dim] * [B, h_num, h_dim, S] -> [B, h_num, S, S] 
        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(self.h_dim)
        
        if mask is not None:
            # [B,S or 1,S] -> [B,1,S or 1,S]
            mask = mask.unsqueeze(1)
            
            # [B, h_num, S, S]
            scores = scores.masked_fill(mask == 0, -1e9)

        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        
        # [B, h_num, S, S] * [B, h_num, S, h_dim] -> [B, h_num, S, h_dim]
        scores = torch.matmul(scores, v)
        return scores
    
    def forward(self, q, k, v, mask=None):
        
        B = q.shape[0]
        
        # Project input and split dimentions
        # [B, S ,model_dim] -> [B, h_num, S, h_dim]
        k = self.k_fc(k).view(B, -1, self.h_num, self.h_dim).transpose(1,2)
        q = self.q_fc(q).view(B, -1, self.h_num, self.h_dim).transpose(1,2)
        v = self.v_fc(v).view(B, -1, self.h_num, self.h_dim).transpose(1,2)
        
        # Apply attention to each head and restore shape
        # -> [B, h_num, S, h_dim]
        attention_scores = self.head_attention(q, k, v, mask)
        
        # [B, h_num, S, h_dim] -> [B, S ,model_dim]
        result = attention_scores.transpose(1,2).contiguous().view(B, -1, self.model_dim)
        result = self.out_fc(result)
    
        return result