import torch
import torch.nn as nn
from tqdm import tqdm
class CNN(nn.Module):

    def __init__(self, num_classes=4, f1=32, f2=64, f3=128, dropout=0.65, chans=1, h=22, w=1000, device=None):
        super(CNN, self).__init__()

        self.device = device

        self.conv1 = nn.Sequential(
            nn.Conv2d(chans, f1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(f1, f2, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(f2, f3, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(f3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout/2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(11776, 1024),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def run_train(self, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=100):
        
        train_losses = []
        test_losses = []
        test_accuracies = []
        train_accuracies = []

        for epoch in range(num_epochs):
            if self.device:
                self.train().to('mps')
            else:
                self.train()

            with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}',
                  position=0, leave=True) as pbar:
                
                avg_loss = 0
                for batch in train_loader:
                    inputs, labels = batch
                    inputs = inputs.float()
                    labels = labels.long()

                    if self.device:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self(inputs.unsqueeze(1))
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    avg_loss += loss

                    if self.device:
                        loss = loss.to(self.device)

                    pbar.update(1)
                    pbar.set_postfix(loss=loss.item())

                train_loss = avg_loss / len(train_loader)
                train_losses.append(train_loss.to('cpu').detach().numpy())
                print(f"Epoch {epoch + 1} - Avg Train Loss: {train_loss:.4f}")
            
            test_loss, test_accuracy, train_accuracy = self.run_eval(train_loader, test_loader, criterion)
            test_losses.append(test_loss.to('cpu'))
            test_accuracies.append(test_accuracy)
            train_accuracies.append(train_accuracy)
            print(f"Epoch {epoch + 1} - Avg Val Loss: {test_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
        scheduler.step()

        return train_losses, test_losses, train_accuracies, test_accuracies
    
    def run_eval(self, train_loader, test_loader, criterion):

        if self.device:
            self.eval().to('mps')
        else:
            self.eval()

        with torch.no_grad():
            test_loss = 0.0
            num_correct_test = 0
            num_samples_test = 0
            num_correct_train = 0
            num_samples_train = 0

            for batch in test_loader:

                inputs, labels = batch
                inputs = inputs.float()
                labels = labels.long()

                if self.device:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                
                outputs = self.forward(inputs.unsqueeze(1))

                if self.device:
                    outputs = outputs.to(self.device)

                test_loss += criterion(outputs, labels)

                _, predictions = torch.max(outputs, dim=1)
                num_correct_test += (predictions == labels).sum().item()
                num_samples_test += len(inputs)

            for batch in train_loader:
                    
                inputs, labels = batch
                inputs = inputs.float()
                labels = labels.long()

                if self.device:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                outputs = self.forward(inputs.unsqueeze(1))

                if self.device:
                    outputs = outputs.to(self.device)

                _, predictions = torch.max(outputs, dim=1)
                num_correct_train += (predictions == labels).sum().item()
                num_samples_train += len(inputs)

        avg_loss = test_loss / len(test_loader)
        accuracy_test = num_correct_test / num_samples_test * 100
        accuracy_train = num_correct_train / num_samples_train * 100

        return avg_loss, accuracy_test, accuracy_train