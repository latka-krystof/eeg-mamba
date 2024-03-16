import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn

train_val_file = "eeg_data/X_train_valid.npy"
train_val_labels_file = "eeg_data/y_train_valid.npy"
train_val_persons_file = "eeg_data/person_train_valid.npy"
test_file = "eeg_data/X_test.npy"
test_labels_file = "eeg_data/y_test.npy"
test_persons_file = "eeg_data/person_test.npy"

class EEGDataset(Dataset):
    def __init__(self, train=True, transform=None, device="cpu"):
        if train:
            self.X = torch.tensor(np.load("eeg_data/X_train_valid.npy")).to(device)
            self.y = torch.tensor(np.load("eeg_data/y_train_valid.npy")).long().to(device)
            self.person_ids = torch.tensor(np.load("eeg_data/person_train_valid.npy")).long().to(device)
        else:
            self.X = torch.tensor(np.load("eeg_data/X_test.npy")).to(device)
            self.y = torch.tensor(np.load("eeg_data/y_test.npy")).long().to(device)
            self.person_ids = torch.tensor(np.load("eeg_data/person_test.npy")).long().to(device)
            
        self.y = self.y - 769
    
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        if self.transform is not None: 
            x = self.transform(x)
        return x, self.y[idx]

class NewEEGDataset(Dataset):
    def __init__(self, samples, labels, persons, device='cpu', mode='train', noise=True):
        self.X = torch.tensor(samples).to(device)
        self.y = torch.tensor(labels).long().to(device)
        self.persons = torch.tensor(persons).long().to(device)
        self.mode = mode

        if self.mode == 'train': 
            
            # Apply data augmentation when loading training set
            total_X = None
            total_y = None

            # indices = np.where(self.persons == 0)[0]

            # self.X = self.X[indices]
            # self.y = self.y[indices]

            # Trimming
            X_trim = self.X[:, :, 0:500]

            # MaxPooling
            X_max = nn.MaxPool1d(kernel_size=2, stride=2)(X_trim)

            total_X = X_max
            total_y = self.y

            # Average + noise
            X_reshaped = X_trim.view(X_trim.shape[0], X_trim.shape[1], -1, 2)
            X_avg = torch.mean(X_reshaped, dim=3)
            X_avg = X_avg + 0.5*torch.randn_like(X_avg)

            total_X = torch.cat((total_X, X_avg), dim=0)
            total_y = torch.cat((total_y, self.y), dim=0)

            # Subsampling with noise
            for i in range(2):
                X_sub = X_trim[:, :, i::2]
                X_sub = X_sub + 0.5*torch.randn_like(X_sub)
                total_X = torch.cat((total_X, X_sub), dim=0)
                total_y = torch.cat((total_y, self.y), dim=0)

            self.X = total_X
            self.y = total_y

        else:

            # indices = np.where(self.persons == 3)[0]
            # self.X = self.X[indices]
            # self.y = self.y[indices]

            # Simple pre-processing to have matching dimensions with training set
            
            # Trimming
            self.X = self.X[:, :, 0:500]
            # MaxPooling
            self.X = nn.MaxPool1d(kernel_size=2, stride=2)(self.X)

        self.y = self.y - 769

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        # Apply any post-processing transforms
        return x, self.y[idx]

def generate_dataloaders(val=0.2, batch_size=64, transform=None):
    train_val_samples = np.load(train_val_file)
    train_val_labels = np.load(train_val_labels_file)
    train_val_persons = np.load(train_val_persons_file)
    test_samples = np.load(test_file)
    test_labels = np.load(test_labels_file)
    test_persons = np.load(test_persons_file)

    split_idx = int(len(train_val_samples) * (1 - val))

    train_samples, val_samples = train_val_samples[:split_idx], train_val_samples[split_idx:]
    train_labels, val_labels = train_val_labels[:split_idx], train_val_labels[split_idx:]
    train_persons, val_persons = train_val_persons[:split_idx], train_val_persons[split_idx:]

    train_dataset = NewEEGDataset(train_samples, train_labels, train_persons, mode='train')
    val_dataset = NewEEGDataset(val_samples, val_labels, val_persons, mode='val')
    test_dataset = NewEEGDataset(test_samples, test_labels, test_persons, mode='test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
