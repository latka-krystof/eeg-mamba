import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from tqdm import tqdm

class MambaEEG(nn.Module):

    def __init__(self, input_dim=22, hidden_dim=24, d_state=16, d_conv=4, dropout=0.5, num_classes=4, device='cuda'):
        super(MambaEEG, self).__init__()
        
        # self.pos_embeddings = PositionEmbedding(250, hidden_dim)
        
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.mamba1 = Mamba(
            d_model=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=25
        )
        self.mamba2 = Mamba(
            d_model=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=25
        )
        # self.mamba3 = Mamba(
        #     d_model=hidden_dim,
        #     d_state=1,
        #     d_conv=16,
        #     expand=2
        # )
        self.ln1 = nn.LayerNorm((250, hidden_dim))
        self.ln2 = nn.LayerNorm((250, hidden_dim))
        # self.ln3 = nn.LayerNorm((250, hidden_dim))
        self.fc = nn.Linear(hidden_dim, num_classes)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # self.dropout3 = nn.Dropout(0.5)

        self.device = device
        
        
    def forward(self, x):
        # print("Input shape: ", x.shape)
        x = torch.transpose(x, 1,2)
        # print("Input shape: ", x.shape)
        
        x = self.embedding(x)
        # x = self.pos_embeddings(x)
        # print("Embed shape: ", x.shape)
        x = self.dropout1(x)
        x = self.mamba1(x)
        
        x = self.ln1(x)
        x = self.dropout2(x)
        x = self.mamba2(x)
        
        x = self.ln2(x)
        
        # x = self.mamba3(x)
        # x = self.dropout3(x)
        # x = self.ln3(x)
        # print("Mamba shape: ", x.shape)
        
        x = x.mean(dim=1)
        # x = x[:, 0, :]
        # print("Before forward shape: ", x.shape)
        x = self.fc(x)
        # print("After forward shape: ", x.shape)
        return x
        
# batch, length, dim = 2, 64, 16
# x = torch.randn(batch, length, dim).to("cuda")
# model = Mamba(
#     # This module uses roughly 3 * expand * d_model^2 parameters
#     d_model=dim, # Model dimension d_model
#     d_state=16,  # SSM state expansion factor
#     d_conv=4,    # Local convolution width
#     expand=2,    # Block expansion factor
# ).to("cuda")
# y = model(x)
# assert y.shape == x.shape
class PositionEmbedding(nn.Module):
    def __init__(self, seq_len, embedding_dim):
        super().__init__()
        self.embed = nn.Embedding(seq_len, embedding_dim)
    def forward(self, x):
        _, T, _ = x.shape
        return x + self.embed(torch.arange(T).to(x.device))