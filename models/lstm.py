import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

class LSTM(nn.Module):
    
    def __init__(self, input_size=22, hidden_size=128, num_layers=5, num_classes=4, dropout=0.4, device='cuda'):
        super(LSTM, self).__init__()

        self.dropout = dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=self.dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.device = device
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        # x = x[:, -1, :]
        x = x.mean(dim=1)
        x = self.fc(F.dropout(x, p=self.dropout))
        return x
