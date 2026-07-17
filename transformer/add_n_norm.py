import torch
from torch import nn

class AddAndNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, initial_x, transformed_x):
        sum = initial_x + transformed_x
        means = torch.mean(sum, keepdim=True, dim=-1)
        stds = torch.std(sum, keepdim=True, dim=-1)
        output = (sum - means) / (stds + 1e-5)
        return output * self.gamma + self.beta