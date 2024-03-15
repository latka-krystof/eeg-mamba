import torch.nn as nn
from training_utils.loops import run_eval, run_train

class CNN_1D(nn.Module):

    def __init__(self, num_classes=4, dropout=0.65, device=None):
        super(CNN_1D, self).__init__()

        self.device = device

        self.conv1 = nn.Sequential(
            nn.Conv1d(22, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=7, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=9, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(5632, 1024),
            nn.ELU(),
            nn.Dropout(dropout)
        )

        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x