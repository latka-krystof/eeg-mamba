import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import wandb

class DepthwiseSeparableConv2D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, depth=1, max_norm=1.0):

        super(DepthwiseSeparableConv2D, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels * depth, kernel_size,
                                        groups=in_channels, padding=(0, kernel_size[1] // 2),
                                        bias=False)
        self.pointwise_conv = nn.Conv2d(in_channels * depth, out_channels, 1, bias=False)
        self.max_norm = max_norm

    def forward(self, x):

        out = self.depthwise_conv(x)
        out = self.pointwise_conv(out)

        with torch.no_grad():
            depthwise_weights = self.depthwise_conv.weight
            pointwise_weights = self.pointwise_conv.weight
            depthwise_norm = torch.norm(depthwise_weights.view(depthwise_weights.size(0), -1), p=2, dim=1)
            pointwise_norm = torch.norm(pointwise_weights.view(pointwise_weights.size(0), -1), p=2, dim=1)
            depthwise_scale = torch.clamp(depthwise_norm / self.max_norm, max=1.0)
            pointwise_scale = torch.clamp(pointwise_norm / self.max_norm, max=1.0)
            self.depthwise_conv.weight.data /= depthwise_scale.view(-1, 1, 1, 1)
            self.pointwise_conv.weight.data /= pointwise_scale.view(-1, 1, 1, 1)

        return out

# Architecture tweeked specifically for BCI data, based on https://arxiv.org/abs/1611.08024
class EEGNet(nn.Module):

    def __init__(self, num_classes=4, chans=22, samples=1000,
                    kernel_length=64, f1=8, d=2, f2=16, dropout=0.5, device='cuda'):
        super(EEGNet, self).__init__()

        self.device = device

        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(chans, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(f1),
            DepthwiseSeparableConv2D(f1, f1 * d, kernel_size=(1, samples // 4)),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout)
        )

        self.block2 = nn.Sequential(
            DepthwiseSeparableConv2D(f1 * d, f2, kernel_size=(1, 16)),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout)
        )

        self.dense = nn.Linear(f2 * (samples // 32 + 1), num_classes)

    def forward(self, x):
        
        x = self.block1(x)
        x = self.block2(x)
        x = torch.flatten(x, start_dim=1)
        x = F.softmax(self.dense(x), dim=-1)
        return x
    
    def run_train(self, train_loader, val_loader, criterion, optimizer, num_epochs=100, wandb=None):
        
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

            if wandb:
                wandb.log({"train_loss": avg_loss/len(train_loader)})
            print(f"Epoch {epoch + 1} - Avg Train Loss: {avg_loss/len(train_loader):.4f}")
            
            val_loss, accuracy = self.run_eval(val_loader, criterion)
            if wandb:
                wandb.log({"val_loss": val_loss, "accuracy": accuracy})
            print(f"Epoch {epoch + 1} - Avg Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")

            # wandb.log({'train_loss': avg_loss/len(train_loader), 'val_loss': val_loss, 'val_accuracy': accuracy})
    
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

    def run_test(self, test_loader, criterion):

        self.eval()
        with torch.no_grad():
            test_loss = 0.0
            num_correct = 0
            num_samples = 0

            for batch in test_loader:

                inputs, labels = batch
                inputs = inputs.float().unsqueeze(1)
                labels = labels.long()
                outputs = self.forward(inputs)
                test_loss += criterion(outputs, labels)

                _, predictions = torch.max(outputs, dim=1)
                num_correct += (predictions == labels).sum().item()
                num_samples += len(inputs)

        avg_loss = test_loss / len(test_loader)
        accuracy = num_correct / num_samples

        return avg_loss, accuracy