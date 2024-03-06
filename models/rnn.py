import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

class RNN(nn.Module):
    
    def __init__(self, input_size=22, hidden_size=128, num_layers=5, num_classes=4, dropout=0.4):
        super(RNN, self).__init__()

        self.dropout = dropout
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=self.dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        x = self.fc(F.dropout(x, p=self.dropout))
        return x

    def run_train(self, train_loader, val_loader, criterion, optimizer, num_epochs=100):
        
        for epoch in range(num_epochs):
            self.train()

            with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}',
                  position=0, leave=True) as pbar:
                
                avg_loss = 0
                for batch in train_loader:
                    inputs, labels = batch
                    inputs = inputs.float().permute(0, 2, 1)
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
                inputs = inputs.float().permute(0, 2, 1)
                labels = labels.long()
                outputs = self.forward(inputs)
                val_loss += criterion(outputs, labels)

                _, predictions = torch.max(outputs, dim=1)
                num_correct += (predictions == labels).sum().item()
                num_samples += len(inputs)

        avg_loss = val_loss / len(val_loader)
        accuracy = num_correct / num_samples

        return avg_loss, accuracy

        