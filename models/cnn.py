import torch
import torch.nn as nn
from tqdm import tqdm
class CNN(nn.Module):

    def __init__(self, num_classes=4, f1=32, f2=64, f3=128, dropout=0.65, chans=1, h=22, w=1000, device='cuda'):
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
            nn.Linear(15744, 1024),
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