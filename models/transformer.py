import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from tqdm import tqdm

class Transformer(nn.Module):

    def __init__(self, input_dim=1000, hidden_dim=4, num_layers=4, num_heads=4, dropout=0.5, num_classes=4, device='cuda'):
        super(Transformer, self).__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layers = TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, 
                                                 dropout=dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.device = device

    def forward(self, x):

        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # Transformer expects shape (seq_len, batch_size, embedding_dim)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # Aggregate over time
        x = self.fc(x)
        return x