import torch
import torch.nn as nn
import torch.nn.functional as F
# from mamba_ssm import Mamba
# from tqdm import tqdm

class MambaEEG(nn.Module):
    pass

#     def __init__(self, input_dim=1000, hidden_dim=128, num_layers=3, num_heads=4, dropout=0.1, num_classes=4):
#         super(MambaEEG, self).__init__()
        
#         self.embedding = nn.Linear(input_dim, hidden_dim)
#         self.mamba = Mamba(
#             d_model=hidden_dim,
#             d_state=16,
#             d_conv=4,
#             expand=2
#         )
#         self.fc = nn.Linear(hidden_dim, num_classes)
        
#     def forward(self, x):
#         x = self.embedding(x)
#         x = self.mamba(x)
#         print(x.shape)
#         x = x.mean(dim=1)
#         x = self.fc(x)
#         return x
        
        
        
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
