import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import numpy as np
import os

class MyDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor, allow_pickle: torch.Tensor):
        self.images = images
        self.labels = labels
        self.allow_pickle = allow_pickle
        
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        
        return [image, label]
    
    def __len__(self):
        return len(self.images)

def mnist(directory: str = 'data/corruptmnist'):
    
    train_datasets = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        data = np.load(f)
        data = dict(zip(("{}".format(item) for item in data), (torch.from_numpy(data[item]) for item in data)))
        dataset = MyDataset(data['images'].to(torch.float32), data['labels'], data['allow_pickle'])
        if 'train' in filename:
            train_datasets.append(dataset)
        else: test = dataset
        
    train = ConcatDataset(train_datasets)
    train = DataLoader(train, batch_size=64, shuffle=True)
    test = DataLoader(test, batch_size=64, shuffle=True)
    
    return train, test