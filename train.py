import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim
from tqdm import tqdm
from dataset import EEGDataset
import numpy as np
from models.mlp import MLP
from models.rnn import RNN
from models.cnn import CNN
from models.lstm import LSTM


def train(experiment_name):
    print("Running experiment: ", experiment_name)
    
    if experiment_name == "cnn":
       
        transform = None
       
        train_dataset = EEGDataset(train=True, transform=transform)
        test_dataset = EEGDataset(train=False, transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
        
        model = CNN()
        optimizer = optim.AdamW(model.parameters(), lr=0.00001)
        criterion = nn.CrossEntropyLoss()
        
        model.run_train(train_loader, test_loader, criterion, optimizer, num_epochs=10)

    elif experiment_name == "rnn":
        
        transform = None
       
        train_dataset = EEGDataset(train=True, transform=transform)
        test_dataset = EEGDataset(train=False, transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)
        
        model = RNN(input_size=22, hidden_size=200, num_layers=2, num_classes=4)
        optimizer = optim.AdamW(model.parameters(), lr=0.00001)
        criterion = nn.CrossEntropyLoss()
        
        model.run_train(train_loader, test_loader, criterion, optimizer, num_epochs=10)
    
    elif experiment_name == "mlp":
        
        transform = None
        
        train_dataset = EEGDataset(train=True, transform=transform)
        test_dataset = EEGDataset(train=False, transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
        
        model = MLP([1000*22, 1000, 100, 4])
        optimizer = optim.AdamW(model.parameters(), lr=0.00001)
        criterion = nn.CrossEntropyLoss()
        
        model.run_train(train_loader, test_loader, criterion, optimizer, num_epochs=10)
        
    elif experiment_name == "lstm":
        transform = None
       
        train_dataset = EEGDataset(train=True, transform=transform)
        test_dataset = EEGDataset(train=False, transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
        
        model = LSTM(input_size=22, hidden_size=100, num_layers=2, num_classes=4, device="cpu")
        optimizer = optim.AdamW(model.parameters(), lr=0.00001)
        criterion = nn.CrossEntropyLoss()
        
        model.run_train(train_loader, test_loader, criterion, optimizer, num_epochs=10)
        
        
        
        
if __name__ == "__main__":
    train("lstm")
    train("rnn")
    train("mlp")
    train("cnn")



# def trainRNN(model, TrainDataLoader, ValDataLodaer, criterion, optimizer, num_epochs=10):
#     for epoch in tqdm(range(num_epochs)):
#         for batch in TrainDataLoader:
#             inputs, labels = batch
#             inputs = inputs.float()
#             labels = labels.float()
#             optimizer.zero_grad()
#             # print(inputs.shape)
#             outputs = model(inputs).squeeze()
#             # print(outputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
        
#         with torch.no_grad():
#             val_loss = 0
#             for batch in ValDataLodaer:
#                 inputs, labels = batch
#                 inputs = inputs.float()
#                 labels = labels.float()
#                 outputs = model(inputs).squeeze()
#                 val_loss += criterion(outputs, labels)
#             print(f"Epoch {epoch} - Val Loss: {val_loss/len(ValDataLodaer)}")
            
            
# if __name__ == "__main__":
    
#     X_test = np.load("eeg_data/X_test.npy")
#     Y_test = np.load("eeg_data/y_test.npy")
#     person_test = np.load("eeg_data/person_test.npy")
    
#     X_train_valid = np.load("eeg_data/X_train_valid.npy")
#     Y_train_valid = np.load("eeg_data/y_train_valid.npy")
#     person_train_valid = np.load("eeg_data/person_train_valid.npy")

#     train_dataset = EEGDataset(X_train_valid, Y_train_valid, person_train_valid)
#     test_dataset = EEGDataset(X_test, Y_test, person_test)

#     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

#     class Flatten(torch.nn.Module):
#         def forward(self, x):
#             batch_size = x.shape[0]
#             return x.view(batch_size, -1)
        
#     class Transpose(torch.nn.Module):
#         def forward(self, x):
#             return x.permute(0, 2, 1)
        
#     class GrabLastElement(torch.nn.Module):
#         def forward(self, x):
#             x = x[-1]
#             print(x[-1].shape)
#             return x[-1]
        

#     model = nn.Sequential(
#         Flatten(),
#         nn.Linear(1000*22, 1000),
#         nn.ReLU(),
#         nn.Linear(1000, 100),
#         nn.ReLU(),
#         nn.Linear(100, 1)
#     )

    
#     # model = nn.Sequential(
#     #     Transpose(),
#     #     nn.LSTM(22, 100, 2, batch_first=True, proj_size=1),
#     #     GrabLastElement(),
#     # )
                          

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.0001)

#     trainMLP(model, train_loader, test_loader, criterion, optimizer, num_epochs=10)
    