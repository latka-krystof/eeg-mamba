import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from tqdm import tqdm

class MambaDepthWiseEEG(nn.Module):
    pass

    def __init__(self, input_dim=22, hidden_dim=128, num_layers=3, num_heads=4, dropout=0.1, num_classes=4):
        super(MambaDepthWiseEEG, self).__init__()
        
        self.embedding = nn.ModuleList([nn.Linear(1, hidden_dim) for i in range(input_dim)])
        self.mambas = nn.ModuleList([Mamba(
            d_model=hidden_dim,
            d_state=16,
            d_conv=4,
            expand=2
        ) for i in range(input_dim)])
        
        # self.fcs = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for i in range(input_dim)])
        
        self.mixer = nn.Sequential(nn.Linear(hidden_dim*input_dim, hidden_dim*input_dim*3), 
                                   nn.ReLU(), 
                                   nn.Linear(hidden_dim*input_dim*3, hidden_dim*input_dim), 
                                   nn.ReLU(), 
                                   nn.Linear(hidden_dim*input_dim, num_classes))
        
        # self.concat_fc = nn.Linear(input_dim*num_classes, num_classes)
        
    def forward(self, x):
        x = torch.transpose(x, 1,2)
        x_mamba = []
        for i, embedding, mamba in zip(range(x.shape[1]), self.embedding, self.mambas):
            # print(x[:, :, i].unsqueeze(2).shape, embedding)
            x_t = embedding(x[:, :, i].unsqueeze(2))
            x_t = mamba(x_t)
            x_t = x_t.mean(dim=1)
            # x_t = x_t[:, -1, :]
            # x_t = fc(x_t)
            x_mamba.append(x_t)
        
        
        # print(x_mamba[0].shape)
        x_mamba = torch.stack(x_mamba, dim=1)
        x = self.mixer(x_mamba.flatten(start_dim=1))
        # print(x.shape)
        # x = x.mean(dim=1)
        # print(x.shape)
        # print(x_mamba.shape)
        
        # x = self.concat_fc(x)
        
        return x
        
    def run_train(self, train_loader, val_loader, criterion, optimizer, num_epochs=100):
        
        for epoch in range(num_epochs):
            self.train()

            with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}',
                  position=0, leave=True) as pbar:
                
                avg_loss = 0
                for batch in train_loader:
                    inputs, labels = batch
                    inputs = inputs.float()
                    labels = labels.long()
                    optimizer.zero_grad()
                    outputs = self.forward(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    avg_loss += loss

                    pbar.update(1)
                    pbar.set_postfix(loss=loss.item())

            print(f"Epoch {epoch + 1} - Avg Train Loss: {avg_loss/len(train_loader):.4f}")
            
            val_loss, accuracy = self.run_eval(val_loader, criterion)
            print(f"Epoch {epoch + 1} - Avg Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    def run_eval(self, val_loader, criterion):

        self.eval()
        with torch.no_grad():
            val_loss = 0.0
            num_correct = 0
            num_samples = 0

            for batch in val_loader:

                inputs, labels = batch
                inputs = inputs.float()
                labels = labels.long()
                outputs = self.forward(inputs)
                val_loss += criterion(outputs, labels)

                _, predictions = torch.max(outputs, dim=1)
                num_correct += (predictions == labels).sum().item()
                num_samples += len(inputs)

        avg_loss = val_loss / len(val_loader)
        accuracy = num_correct / num_samples

        return avg_loss, accuracy
        
        
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
