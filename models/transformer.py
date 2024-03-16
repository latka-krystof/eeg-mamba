import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from tqdm import tqdm

class Transformer(nn.Module):

    def __init__(self, input_dim=22, hidden_dim=24, num_layers=2, num_heads=4, dropout=0.5, num_classes=4, device='cuda'):
        super(Transformer, self).__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)
        # self.pos_embeddings = PositionEmbedding(250, hidden_dim)
        encoder_layers = TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, 
                                                 dropout=dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.device = device

    def forward(self, x):
        # X.shape = batch_size, channels, seq_len
        x = x.permute(0, 2, 1) 
        
        # X.shape = batch_size, seq_len, channels
        x = self.embedding(x)
        # x = self.pos_embeddings(x)
        x = x.permute(1, 0, 2) 
        
        # X.shape = seq_len, batch_size, channels
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # Aggregate over time
        # x = x[-1, :, :]
        x = self.fc(x)
        return x
    
class PositionEmbedding(nn.Module):
    def __init__(self, seq_len, embedding_dim):
        super().__init__()
        self.embed = nn.Embedding(seq_len, embedding_dim)
    def forward(self, x):
        _, T, _ = x.shape
        return x + self.embed(torch.arange(T).to(x.device))