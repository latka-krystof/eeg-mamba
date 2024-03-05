import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self, layers, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        self.relu = nn.ReLU()
        self.dropout = dropout

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        for layer in self.layers:
            x = self.relu(layer(x))
            x = F.dropout(x, self.dropout)
        return x
    
    def run_train(self, train_loader, val_loader, criterion, optimizer, num_epochs=10):
        for epoch in range(num_epochs):
            avg_loss = 0
            for batch in train_loader:
                inputs, labels = batch
                inputs = inputs.float()
                labels = labels.float()
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                avg_loss += loss
            print(f"Epoch {epoch} - Avg Train Loss: {avg_loss/len(train_loader)}")
            
            with torch.no_grad():
                self.eval()
                val_loss = 0
                for batch in val_loader:
                    inputs, labels = batch
                    inputs = inputs.float()
                    labels = labels.float()
                    outputs = self.forward(inputs).squeeze()
                    val_loss += criterion(outputs, labels)
                self.train()
                print(f"Epoch {epoch} - Val Loss: {val_loss/len(val_loader)}")