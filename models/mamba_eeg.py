import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from tqdm import tqdm

class MambaEEG(nn.Module):
    pass

    def __init__(self, input_dim=22, hidden_dim=128, num_layers=3, num_heads=4, dropout=0.1, num_classes=4):
        super(MambaEEG, self).__init__()
        
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.mamba = Mamba(
            d_model=hidden_dim,
            d_state=16,
            d_conv=4,
            expand=2
        )
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # print("Input shape: ", x.shape)
        x = torch.transpose(x, 1,2)
        x = self.embedding(x)
        # print("Embed shape: ", x.shape)
        x = self.mamba(x)
        # print("Mamba shape: ", x.shape)
        x = x.mean(dim=1)
        # x = x[:, -1, :]
        # print("Before forward shape: ", x.shape)
        x = self.fc(x)
        # print("After forward shape: ", x.shape)
        return x
        
    def run_train(self, train_loader, val_loader, criterion, optimizer, num_epochs=100):
        
        for epoch in range(num_epochs):
            self.train()

            # with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}',
            #       position=0, leave=True) as pbar:
                
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

                # pbar.update(1)
                # pbar.set_postfix(loss=loss.item())

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
