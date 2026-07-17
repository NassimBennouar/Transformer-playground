import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model) # shape : (max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, 1).unsqueeze(1) # shape : (max_seq_len, 1)
        _2i = torch.arange(0, d_model, 2).unsqueeze(0) # shape : (1, d_model)

        denominateur = 10000 ** (_2i/d_model)
        angle = pos / denominateur

        pe[:, 0::2] = torch.sin(angle)
        pe[:, 1::2] = torch.cos(angle)

        pe=pe.unsqueeze(0)

        self.register_buffer('positional_encoding', pe)
        # We need to register as buffer so it's transfered to device (stored in state_dict, not trainable)

    def forward(self, x): # x.shape -> (batch_size, seq_len, d_model)
        print(x.shape)
        print(self.positional_encoding.shape)
        seq_len=x.shape[1]
        x = x + self.positional_encoding[:, :seq_len, :]
        return x

if __name__ == "__main__":
    pe = PositionalEncoding(max_seq_len=2, d_model=4)
    dummy_embeddings = torch.zeros(1, 2, 4)
    print(dummy_embeddings)
    output = pe(dummy_embeddings)
    print(output.shape)
    print(output)
    print()
    print(torch.arange(0, 6, 2).unsqueeze(0))
