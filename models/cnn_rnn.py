import torch.nn as nn
from training_utils.loops import run_eval, run_train

class CNN_RNN(nn.Module):

    def __init__(self, num_classes=4, dropout=0.65, rnn_dropout=0.4, hidden_size=128, num_layers=1, device='cuda'):
        super(CNN_RNN, self).__init__()

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

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

        self.gru = nn.GRU(11, hidden_size, num_layers, batch_first=True)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x