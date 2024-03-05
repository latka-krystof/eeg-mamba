import torch
from torch.utils.data import Dataset
import numpy as np

class EEGDataset(Dataset):
    def __init__(self, train=True, transform=None):
        if train:
            self.X = torch.tensor(np.load("eeg_data/X_train_valid.npy"))
            self.y = torch.tensor(np.load("eeg_data/y_train_valid.npy")).long()
            self.person_ids = torch.tensor(np.load("eeg_data/person_train_valid.npy")).long()
        else:
            self.X = torch.tensor(np.load("eeg_data/X_test.npy"))
            self.y = torch.tensor(np.load("eeg_data/y_test.npy")).long()
            self.person_ids = torch.tensor(np.load("eeg_data/person_test.npy")).long()
        self.y = self.y - 769
        # one hot encode 0,1,2,3
        self.y = torch.nn.functional.one_hot(self.y, num_classes=4)
    
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        if self.transform is not None: 
            x = self.transform(x)
        return x, self.y[idx]

# def get_train_and_test():
   
    
#     return EEGDataset(X_train_valid, Y_train_valid, person_train_valid), EEGDataset(X_test, Y_test, person_test)



# import numpy as np
# X_test = np.load("eeg_data/X_test.npy")
# Y_test = np.load("eeg_data/y_test.npy")
# person_test = np.load("eeg_data/person_test.npy")

# X_train_valid = np.load("eeg_data/X_train_valid.npy")
# Y_train_valid = np.load("eeg_data/y_train_valid.npy")
# person_train_valid = np.load("eeg_data/person_train_valid.npy")

# print("X_test shape: ", X_test.shape)
# print("Y_test shape: ", Y_test.shape)
# print("person_test shape: ", person_test.shape)
# print(Y_test-769)

# print()

# print("X_train_valid shape: ", X_train_valid.shape)
# print("Y_train_valid shape: ", Y_train_valid.shape)
# print("person_train_valid shape: ", person_train_valid.shape)