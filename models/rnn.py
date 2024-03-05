import torch
import torch.nn as nn

class RNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.rnn(x)
        x = self.fc(x[:, -1, :])

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
        