import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

class CNN(nn.Module):

    def __init__(self, num_classes=4):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 4 * 248, 1000)
        self.fc2 = nn.Linear(1000, 120)
        self.fc3 = nn.Linear(120, 84)
        self.fc4 = nn.Linear(84, num_classes)
    
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 248)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
    def run_train(self, train_loader, val_loader, criterion, optimizer, num_epochs=10):
        
        for epoch in range(num_epochs):
            self.train()

            with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}',
                  position=0, leave=True) as pbar:
                
                avg_loss = 0
                for batch in train_loader:
                    inputs, labels = batch
                    inputs = inputs.float().unsqueeze(1)
                    labels = labels.long()
                    optimizer.zero_grad()
                    outputs = self.forward(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    avg_loss += loss

                    pbar.update(1)
                    pbar.set_postfix(loss=loss.item())

                print(f"Epoch {epoch} - Avg Train Loss: {avg_loss/len(train_loader):.4f}")
            
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
                inputs = inputs.float().unsqueeze(1)
                labels = labels.long()
                outputs = self.forward(inputs)
                val_loss += criterion(outputs, labels)

                _, predictions = torch.max(outputs, dim=1)
                num_correct += (predictions == labels).sum().item()
                num_samples += len(inputs)

        avg_loss = val_loss / len(val_loader)
        accuracy = num_correct / num_samples

        return avg_loss, accuracy
