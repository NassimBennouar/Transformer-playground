from torch import nn
from transformer.attention import MultiHeadAttention
from transformer.add_n_norm import AddAndNorm
from transformer.ffn import FFN

class Encoder(nn.Module):
    def __init__(self, h, d_model):
        super().__init__()
        self.self_attn = MultiHeadAttention(h, d_model)
        self.add_n_norm1 = AddAndNorm(d_model)
        self.add_n_norm2 = AddAndNorm(d_model)
        self.ffn = FFN(d_model)

    def forward(self, x, mask = None):
        sublayer = self.self_attn(x, padding_mask=mask)
        added = self.add_n_norm1(x, sublayer)
        sublayer2 = self.ffn(added)
        return self.add_n_norm2(added, sublayer2)