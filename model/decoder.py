from torch import nn
from model.attention import MultiHeadAttention
from model.utils import FeedForward, clone, PositionalEncoding


class Decoder(nn.Module):
    def __init__(self, model_dim, heads_num):
        super().__init__()
        self.lnorm_1 = nn.LayerNorm(model_dim)
        self.attention_1 = MultiHeadAttention(heads_num, model_dim)
        self.dropout_1 = nn.Dropout(0.1)
        
        self.lnorm_2 = nn.LayerNorm(model_dim)
        self.attention_2 = MultiHeadAttention(heads_num, model_dim)
        self.dropout_2 = nn.Dropout(0.1)
        
        self.lnorm_3 = nn.LayerNorm(model_dim)
        self.ff = FeedForward(model_dim)
        self.dropout_3 = nn.Dropout(0.1)
        
    def connection(self, x, sublayer, norm, drop):
        y = norm(x)
        y = drop(sublayer(y))
        return x + y
        
    def forward(self, x, e_out, src_msk, trg_msk):
        x = self.connection(x, lambda x: self.attention_1(x, x, x, trg_msk),
                            self.lnorm_1, self.dropout_1)
        x = self.connection(x, lambda x: self.attention_1(x, e_out, e_out, src_msk),
                            self.lnorm_2, self.dropout_2)
        x = self.connection(x, self.ff,
                            self.lnorm_3, self.dropout_3)
        return x

class DencoderStack(nn.Module):
    def __init__(self, vocab_size, model_dim, N, heads_num):
        super().__init__()
        self.N = N
        self.emb = nn.Embedding(vocab_size, model_dim)
        self.pe = PositionalEncoding(model_dim)
        self.layers = clone(Decoder(model_dim, heads_num), N)
        self.norm = nn.LayerNorm(model_dim)
        
    def forward(self, x, e_out, src_msk, trg_msk):
        x = self.emb(x)
        x = self.pe(x)
        for l_idx in range(self.N):
            x = self.layers[l_idx](x, e_out, src_msk, trg_msk)
        return self.norm(x)