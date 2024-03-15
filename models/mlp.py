import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self, layers, dropout=0.5, device='cuda'):
        super().__init__()

        self.layers = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        self.dropout = dropout
        self.device = device

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        for layer in self.layers:
            x = F.relu(layer(x))
            x = F.dropout(x, self.dropout)
        x = F.softmax(x, dim=-1)
        return x