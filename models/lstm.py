import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

class LSTM(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super(LSTM, self).__init__()
        self.device = device
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, dropout=0.5, batch_first=True).to(device)
        self.fc = nn.Linear(hidden_size*2, num_classes).to(device)
        
    
    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.rnn(x)
        x = x[0]
        x = x[:, -1, :].squeeze()
        # print(x.shape)
        x = self.fc(x)
        # print(x.shape)
        return x

    def run_train(self, train_loader, val_loader, criterion, optimizer, num_epochs=10):
        
        for epoch in range(num_epochs):
            self.train()

            with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}',
                  position=0, leave=True) as pbar:
                
                avg_loss = 0
                for batch in train_loader:
                    inputs, labels = batch
                    
                    inputs = inputs.to(torch.float32)
                    labels = labels.long()
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.forward(inputs)
                    # print(outputs.shape, labels.shape)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    avg_loss += loss

                    pbar.update(1)
                    pbar.set_postfix(loss=loss.item())

                print(f"Epoch {epoch} - Avg Train Loss: {avg_loss/len(train_loader):.4f}\n")
            
            val_loss, accuracy = self.run_eval(val_loader, criterion)
            print(f"Epoch {epoch + 1} - Avg Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}\n")
    
    def run_eval(self, val_loader, criterion):

        self.eval()
        with torch.no_grad():
            val_loss = 0.0
            num_correct = 0
            num_samples = 0

            for batch in val_loader:

                inputs, labels = batch
                inputs = inputs.to(torch.float32)
                labels = labels.long()
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.forward(inputs)
                val_loss += criterion(outputs, labels)

                _, predictions = torch.max(outputs, dim=1)
                num_correct += (predictions == labels).sum().item()
                num_samples += len(inputs)

        avg_loss = val_loss / len(val_loader)
        accuracy = num_correct / num_samples

        return avg_loss, accuracy