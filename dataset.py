'''
Provides the dataloader for the recurrent datafiles
Due to the custom RNN and how it is impelemented, the 'batch' option in the dataloader can not be used
The batchsize is provided within the datastructure
'''
import torch.multiprocessing
from torch.utils.data import Dataset, DataLoader
import torch

class retraDataset(Dataset):
    def __init__(self, dataPath):
        self.samples = torch.load(dataPath)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,idx):
        return self.samples[idx]