import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

class CNN(nn.Module):

    def __init__(self, num_classes=4, f1=32, f2=64, f3=128, dropout=0.2, chans=22, samples=1000):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(chans, f1, kernel_size=3, padding='same'),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(f1),
            nn.Dropout(dropout)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(f1, f2, kernel_size=3, padding='same'),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(f2),
            nn.Dropout(dropout)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(f2, f3, kernel_size=3, padding='same'),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(f3),
            nn.Dropout(dropout)
        )
        self.fc1 = nn.Linear(f3 * (samples // 8), 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, num_classes)

        self.dropout = dropout
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = torch.flatten(x, start_dim=1)
        x = F.dropout(F.relu(self.fc1(x)), p=self.dropout) 
        x = F.dropout(F.relu(self.fc2(x)), p=self.dropout)   
        x = F.dropout(F.relu(self.fc3(x)), p=self.dropout) 
        x = F.dropout(F.relu(self.fc4(x)), p=self.dropout) 
        x = F.softmax(x, dim=-1)
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
