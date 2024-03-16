import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
from dataset import EEGDataset, NewEEGDataset, generate_dataloaders
import numpy as np
from models.mlp import MLP
from models.rnn import RNN
from models.cnn import CNN
from models.lstm import LSTM
from models.eeg_net import EEGNet
from models.gru import GRU
from models.transformer import Transformer
from models.mamba_eeg import MambaEEG
from models.cnn_1d import CNN_1D
from models.cnn_rnn import CNN_RNN
from models.resnet_1d import ResNet_1D
from data_utils.timeseries_transforms import Spectrogram
from data_utils.timeseries_transforms import Composite, Trimming, MaxPooling, GaussianNoise
from training_utils.loops import run_train, run_testing
import wandb
import matplotlib.pyplot as plt

def train(experiment_name, num_epochs, batch_size, lr, device, wandb_track):

    if wandb_track:
        wandb.init(
            # set the wandb project where this run will be logged
            project="c147_eeg_testing",
            name=experiment_name+"_"+str(lr),
            # track hyperparameters and run metadata
            config={
            "learning_rate": lr,
            "architecture": experiment_name,
            "batch_size": batch_size,
            "num_epochs": num_epochs
            }
        )

    if experiment_name == "mlp":

        train_loader, val_loader, test_loader = generate_dataloaders(
            val=0.1, batch_size=batch_size
        )
       
        model = MLP([250 * 22, 1024, 128, 4], device=device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-1)
        scheduler = StepLR(optimizer, step_size=25, gamma=0.5)

        train_losses, val_losses, train_accuracies, val_accuracies = run_train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=num_epochs, progress_bar=True, progress=True, wandb_track=wandb_track)

        test_accuracy = run_testing(model, test_loader, criterion)
        print(f"Test accuracy: {test_accuracy:.2f}%")

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accuracies": train_accuracies,
            "val_accuracies": val_accuracies
        }

    elif experiment_name == "cnn":

        train_loader, val_loader, test_loader = generate_dataloaders(
            val=0.1, batch_size=batch_size
        )

        criterion = nn.CrossEntropyLoss()
        model = CNN(device=device)

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-1)
        scheduler = StepLR(optimizer, step_size=25, gamma=0.5)

        train_losses, val_losses, train_accuracies, val_accuracies = run_train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=num_epochs, progress_bar=True, progress=True, wandb_track=wandb_track)

        test_accuracy = run_testing(model, test_loader, criterion)
        print(f"Test accuracy: {test_accuracy:.2f}%")

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accuracies": train_accuracies,
            "val_accuracies": val_accuracies
        }
    
    elif experiment_name == "cnn_1d":
        
        train_loader, val_loader, test_loader = generate_dataloaders(
            val=0.1, batch_size=batch_size
        )

        model = CNN_1D(device=device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-1)
        scheduler = StepLR(optimizer, step_size=25, gamma=0.5)

        train_losses, val_losses, train_accuracies, val_accuracies = run_train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=num_epochs, progress_bar=True, progress=True, wandb_track=wandb_track)

        test_accuracy = run_testing(model, test_loader, criterion)
        print(f"Test accuracy: {test_accuracy:.2f}%")

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accuracies": train_accuracies,
            "val_accuracies": val_accuracies
        }

    elif experiment_name == "cnn_rnn":
        
        train_loader, val_loader, test_loader = generate_dataloaders(
            val=0.1, batch_size=batch_size
        )

        model = CNN_RNN(device=device, hidden_size=128, dropout=0.6)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.008)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.3)

        train_losses, val_losses, train_accuracies, val_accuracies = run_train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=num_epochs, progress_bar=True, progress=True, wandb_track=wandb_track)

        test_accuracy = run_testing(model, test_loader, criterion)
        print(f"Test accuracy: {test_accuracy:.2f}%")

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accuracies": train_accuracies,
            "val_accuracies": val_accuracies
        }

    elif experiment_name == "resnet_1d":
                    
        train_loader, val_loader, test_loader = generate_dataloaders(
            val=0.1, batch_size=batch_size
        )

        model = ResNet_1D(device=device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-1)
        scheduler = StepLR(optimizer, step_size=25, gamma=0.1)

        train_losses, val_losses, train_accuracies, val_accuracies = run_train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=num_epochs, wandb_track=wandb_track)

        test_accuracy = run_testing(model, test_loader, criterion)
        print(f"Test accuracy: {test_accuracy:.2f}%")

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accuracies": train_accuracies,
            "val_accuracies": val_accuracies
        }


    elif experiment_name == "rnn":

        train_loader, val_loader, test_loader = generate_dataloaders(
            val=0.1, batch_size=batch_size
        )

        model = RNN(device=device)
        criterion = nn.CrossEntropyLoss()

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-1)
        scheduler = StepLR(optimizer, step_size=25, gamma=0.1)

        train_losses, val_losses, train_accuracies, val_accuracies = run_train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=num_epochs, wandb_track=wandb_track)

        test_accuracy = run_testing(model, test_loader, criterion)
        print(f"Test accuracy: {test_accuracy:.2f}%")

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accuracies": train_accuracies,
            "val_accuracies": val_accuracies
        }

    elif experiment_name == "lstm":

        train_loader, val_loader, test_loader = generate_dataloaders(
            val=0.1, batch_size=batch_size
        )

        model = LSTM(device=device)
        criterion = nn.CrossEntropyLoss()

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-1)
        scheduler = StepLR(optimizer, step_size=25, gamma=0.1)

        train_losses, val_losses, train_accuracies, val_accuracies = run_train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=num_epochs, wandb_track=wandb_track)

        test_accuracy = run_testing(model, test_loader, criterion)
        print(f"Test accuracy: {test_accuracy:.2f}%")

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accuracies": train_accuracies,
            "val_accuracies": val_accuracies
        }

    elif experiment_name == "eegnet":

        train_loader, val_loader, test_loader = generate_dataloaders(
            val=0.1, batch_size=batch_size
        )

        criterion = nn.CrossEntropyLoss()        
        model = EEGNet(device=device, samples=250)

        optimizer = optim.AdamW(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=25, gamma=0.1)

        train_losses, val_losses, train_accuracies, val_accuracies = run_train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=num_epochs, wandb_track=wandb_track)

        test_accuracy = run_testing(model, test_loader, criterion)
        print(f"Test accuracy: {test_accuracy:.2f}%")

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accuracies": train_accuracies,
            "val_accuracies": val_accuracies
        }

    elif experiment_name == "gru":

        train_loader, val_loader, test_loader = generate_dataloaders(
            val=0.1, batch_size=batch_size
        )

        model = GRU(device=device)
        criterion = nn.CrossEntropyLoss()

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-1)
        scheduler = StepLR(optimizer, step_size=25, gamma=0.1)

        train_losses, val_losses, train_accuracies, val_accuracies = run_train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=num_epochs, wandb_track=wandb_track)

        test_accuracy = run_testing(model, test_loader, criterion)
        print(f"Test accuracy: {test_accuracy:.2f}%")

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accuracies": train_accuracies,
            "val_accuracies": val_accuracies
        }

    elif experiment_name == "transformer":

        train_loader, val_loader, test_loader = generate_dataloaders(
            val=0.1, batch_size=batch_size
        )

        model = Transformer(device=device)
        criterion = nn.CrossEntropyLoss()

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-1)
        scheduler = StepLR(optimizer, step_size=25, gamma=0.1)

        train_losses, val_losses, train_accuracies, val_accuracies = run_train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=num_epochs, wandb_track=wandb_track)

        test_accuracy = run_testing(model, test_loader, criterion)
        print(f"Test accuracy: {test_accuracy:.2f}%")

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accuracies": train_accuracies,
            "val_accuracies": val_accuracies
        }
        
    elif experiment_name == "mamba":

        train_loader, val_loader, test_loader = generate_dataloaders(
            val=0.1, batch_size=batch_size
        )

        model = MambaEEG(device=device)
        criterion = nn.CrossEntropyLoss()

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-1)
        scheduler = StepLR(optimizer, step_size=25, gamma=0.1)

        train_losses, val_losses, train_accuracies, val_accuracies = run_train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=num_epochs, wandb_track=wandb_track)

        test_accuracy = run_testing(model, test_loader, criterion)
        print(f"Test accuracy: {test_accuracy:.2f}%")

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accuracies": train_accuracies,
            "val_accuracies": val_accuracies
        }


# In the hyperparam_sweep function, hardcode the architecture and params to tune
def hyperparam_sweep(config=None):

    wandb.init(project='c147_eeg_testing', config=config)

    batch_size = wandb.config.batch_size
    lr = wandb.config.lr
    weight_decay = wandb.config.weight_decay
    epochs = wandb.config.epochs
    step_size = wandb.config.step_size
    gamma = wandb.config.gamma
    dropout = wandb.config.dropout

    transform = None
    
    train_loader, val_loader, test_loader = generate_dataloaders(
        val=0.1, batch_size=batch_size, transform=transform
    )

    model = CNN_1D(dropout=dropout, device='cuda')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    train_losses, val_losses, train_accuracies, val_accuracies = run_train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=epochs, unsqueeze=False, progress_bar=True, progress=True, sweep=True)

    test_accuracy = run_testing(model, test_loader, criterion, unsqueeze=False)
    print(f"Test accuracy: {test_accuracy:.2f}%")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies
    }
        
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train EEG classification models')
    parser.add_argument('experiment', type=str, help='Name of the experiment/model to train')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs for training (default: 10)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate for optimizer (default: 0.00001)')
    parser.add_argument('--device', type=str, default="cuda", help='Apply data transformations (default: cpu)')
    parser.add_argument('--wandb_track', action="store_true", help='Plot graphs in wandb (default: False)')
    
    args = parser.parse_args()

    if args.experiment == 'sweep':

        # manually change configurations to run a different sweep
        sweep_configuration = {
            "method": "bayes",
            "metric": {"goal": "maximize", "name": "val_accuracy"},
            "parameters": {
                "batch_size": {"values": [32, 64, 128]},
                "lr": {"min": 0.000001, "max": 0.001, "distribution": "uniform"},
                "weight_decay": {"min": 0.001, "max": 0.01, "distribution": "uniform"},
                "epochs": {"values": [60, 80, 100, 120]},
                "step_size": {"values": [10, 20, 30]},
                "gamma": {"values": [0.1, 0.2, 0.3]},
                "dropout": {"values": [0.5, 0.6, 0.7, 0.8]}
            },
            "early_terminate": {
                "type": "hyperband",
                "s": 2, 
                "eta": 3,
                "max_iter": 27
            }
        }

        wandb.login()
        sweep_id = wandb.sweep(sweep_configuration, project='c147_eeg_testing')
        wandb.agent(sweep_id, function=hyperparam_sweep)

    else:
    
        stats = train(args.experiment, args.num_epochs, args.batch_size, args.lr, args.device, args.wandb_track)
        
        if stats:
            train_losses = stats["train_losses"]
            val_losses = stats["val_losses"]
            train_accuracies = stats["train_accuracies"]
            val_accuracies = stats["val_accuracies"]

            plt.figure()
            plt.plot(train_losses, label="Average train Loss")
            plt.plot(val_losses, label="Average validation Loss")
            plt.xlabel("Epoch")
            plt.legend()
            plt.show()

            plt.figure()
            plt.plot(train_accuracies, label="Train Accuracy")
            plt.plot(val_accuracies, label="Validation Accuracy")
            plt.xlabel("Epoch")
            plt.legend()
            plt.show()