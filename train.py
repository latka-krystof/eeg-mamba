import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim
from dataset import EEGDataset
import numpy as np
from models.mlp import MLP
from models.rnn import RNN
from models.cnn import CNN
from models.lstm import LSTM
from models.eeg_net import EEGNet
from models.gru import GRU
from models.transformer import Transformer

def train(experiment_name, num_epochs, batch_size, lr, transforms):

    if experiment_name == "mlp":

        if transforms:
            pass # apply transforms associated with specific model
        else:
            transform = None

        train_dataset = EEGDataset(train=True, transform=transform)
        test_dataset = EEGDataset(train=False, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        criterion = nn.CrossEntropyLoss()
       
        model = MLP([1000 * 22, 1024, 128, 4])
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        model.run_train(train_loader, test_loader, criterion, optimizer, num_epochs=num_epochs)

    elif experiment_name == "cnn":

        if transforms:
            pass # apply transforms associated with specific model
        else:
            transform = None

        train_dataset = EEGDataset(train=True, transform=transform)
        test_dataset = EEGDataset(train=False, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        criterion = nn.CrossEntropyLoss()
        
        model = CNN()
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        model.run_train(train_loader, test_loader, criterion, optimizer, num_epochs=num_epochs)

    elif experiment_name == "rnn":

        if transforms:
            pass # apply transforms associated with specific model
        else:
            transform = None

        train_dataset = EEGDataset(train=True, transform=transform)
        test_dataset = EEGDataset(train=False, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        criterion = nn.CrossEntropyLoss()

        model = RNN()
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        model.run_train(train_loader, test_loader, criterion, optimizer, num_epochs=num_epochs)

    elif experiment_name == "lstm":

        if transforms:
            pass # apply transforms associated with specific model
        else:
            transform = None

        train_dataset = EEGDataset(train=True, transform=transform)
        test_dataset = EEGDataset(train=False, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        criterion = nn.CrossEntropyLoss()
        
        model = LSTM()
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        model.run_train(train_loader, test_loader, criterion, optimizer, num_epochs=num_epochs)

    elif experiment_name == "eegnet":

        if transforms:
            pass # apply transforms associated with specific model
        else:
            transform = None

        train_dataset = EEGDataset(train=True, transform=transform)
        test_dataset = EEGDataset(train=False, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        criterion = nn.CrossEntropyLoss()
        
        model = EEGNet()
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        model.run_train(train_loader, test_loader, criterion, optimizer, num_epochs=num_epochs)

    elif experiment_name == "gru":

        if transforms:
            pass # apply transforms associated with specific model
        else:
            transform = None

        train_dataset = EEGDataset(train=True, transform=transform)
        test_dataset = EEGDataset(train=False, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        criterion = nn.CrossEntropyLoss()
        
        model = GRU()
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        model.run_train(train_loader, test_loader, criterion, optimizer, num_epochs=num_epochs)

    elif experiment_name == "transformer":

        if transforms:
            pass # apply transforms associated with specific model
        else:
            transform = None

        train_dataset = EEGDataset(train=True, transform=transform)
        test_dataset = EEGDataset(train=False, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        criterion = nn.CrossEntropyLoss()
        
        model = Transformer()
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        model.run_train(train_loader, test_loader, criterion, optimizer, num_epochs=num_epochs)
        
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train EEG classification models')
    parser.add_argument('experiment', type=str, help='Name of the experiment/model to train')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training (default: 10)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate for optimizer (default: 0.00001)')
    parser.add_argument('--transforms', action='store_true', help='Apply data transformations (default: False)')
    args = parser.parse_args()
    
    train(args.experiment, args.num_epochs, args.batch_size, args.lr, args.transforms)
    