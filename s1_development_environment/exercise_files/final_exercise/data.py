import numpy as np
import torch


def mnist():
    folder_path = "./../../../data/corruptmnist"    
    train = torch.randn(50000, 784)
    test = torch.randn(10000, 784) 
    path = '/tmp/data/mnist.npz'

    with np.load(path) as f:
        
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        return train, test

    def load_data(path):
        with np.load(path) as f:
            x_train, y_train = f['x_train'], f['y_train']
            x_test, y_test = f['x_test'], f['y_test']
            return (x_train, y_train), (x_test, y_test)
