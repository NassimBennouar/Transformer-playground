import torch.nn as nn
from positional_encoding import PositionalEncoding

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        # nn.Embedding is basically a lookup table initialized with random values,
        # its memory location gets updated throughout training
        self.pe = PositionalEncoding(max_seq_len=max_seq_len, d_model=d_model)

    def forward(self, x): # x.shape (batch_size, seq_len) where each element is a token ID
        x = self.embedding(x)
        x = self.pe(x)
        return x