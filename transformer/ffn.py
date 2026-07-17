import torch
from torch import nn

class FFN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.first_linear=nn.Linear(d_model, 4*d_model)
        self.second_linear=nn.Linear(4*d_model, d_model)

    def forward(self, x):
        hidden = self.first_linear(x)
        hidden = torch.relu(hidden)
        return self.second_linear(hidden)