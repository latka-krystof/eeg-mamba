import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from tqdm import tqdm

class Transformer(nn.Module):

    def __init__(self, input_dim=1000, hidden_dim=128, num_layers=3, num_heads=4, dropout=0.1, num_classes=4):
        super(Transformer, self).__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layers = TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, 
                                                 dropout=dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):

        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # Transformer expects shape (seq_len, batch_size, embedding_dim)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # Aggregate over time
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
    
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