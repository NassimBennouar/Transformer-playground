from torch import nn
from transformer.attention import MultiHeadAttention
from transformer.add_n_norm import AddAndNorm
from transformer.ffn import FFN

class Decoder(nn.Module):
    def __init__(self, h, d_model):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(h, d_model)
        self.add_n_norm1 = AddAndNorm(d_model)
        self.cross_attn = MultiHeadAttention(h, d_model)
        self.add_n_norm2 = AddAndNorm(d_model)
        self.ffn = FFN(d_model)
        self.add_n_norm3 = AddAndNorm(d_model)

    def forward(self, tgt, memory, causal_mask, tgt_padding_mask=None, memory_padding_mask=None):
        # askip, selon claude, la convention pour ce qui sort de l'encodeur
        # c'est memory et pour les output embeddings c'est tgt
        sublayer = self.masked_self_attn(tgt, padding_mask=tgt_padding_mask, causal_mask=causal_mask)
        added = self.add_n_norm1(tgt, sublayer)
        sublayer2 = self.cross_attn(added, memory, padding_mask=memory_padding_mask)
        added2 = self.add_n_norm2(added, sublayer2)
        sublayer3 = self.ffn(added2)
        return self.add_n_norm3(added2, sublayer3)