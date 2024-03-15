import torch.nn as nn
from training_utils.loops import run_eval, run_train

class ResBlock_1D(nn.Module):
    
        def __init__(self, in_channels, out_channels, stride=1, downsample=None):
            super(ResBlock_1D, self).__init__()
    
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.65)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm1d(out_channels)
            self.downsample = downsample
    
        def forward(self, x):
            residual = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.dropout(out)
            out = self.conv2(out)
            out = self.bn2(out)
            if self.downsample:
                residual = self.downsample(x)
            out += residual
            out = self.relu(out)
            return out

class ResNet_1D(nn.Module):

    def __init__(self, num_classes=4, dropout=0.65, device = None):
        super(ResNet_1D, self).__init__()

        self.device = device

        self.conv1 = nn.Sequential(
            nn.Conv1d(22, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self.make_layer(64, 64, 2)
        self.layer2 = self.make_layer(64, 128, 2, stride=2)
        self.layer3 = self.make_layer(128, 256, 2, stride=2)
        self.layer4 = self.make_layer(256, 512, 2, stride=2)
        self.layer5 = self.make_layer(512, 1024, 2, stride=2)

        self.fc = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        downsample = None
        if (stride != 1) or (in_channels != out_channels):
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm1d(out_channels)
            )
        layers = []
        layers.append(ResBlock_1D(in_channels, out_channels, stride, downsample))
        for i in range(1, num_blocks):
            layers.append(ResBlock_1D(out_channels, out_channels))
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = nn.functional.adaptive_avg_pool1d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
