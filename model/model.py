from torch import nn
from model.encoder import EncoderStack
from model.decoder import DencoderStack

class Transformer(nn.Module):
    def __init__(self, src_voc, trg_voc, model_dim, N, heads_num):
        super().__init__()
        self.encoder = EncoderStack(src_voc, model_dim, N, heads_num)
        self.decoder = DencoderStack(trg_voc, model_dim, N, heads_num)
        self.out_fc = nn.Linear(model_dim, trg_voc)
        
    def forward(self, x, y, src_msk, trg_msk):
        e_out = self.encoder(x, src_msk)
        d_out = self.decoder(y, e_out, src_msk, trg_msk)
        out = self.out_fc(d_out)
        return out