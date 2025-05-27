from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
# def data_loaders(batch_size, data_dir,shuffle=True):
#     train_loader = DataLoader(
#         MNIST(root = data_dir, train=True, download=True,
#                        transform=transforms.Compose([
#                            transforms.ToTensor()
#                        ])),
#         batch_size=batch_size)

#     test_loader = DataLoader(
#         MNIST(root = data_dir, train=False, transform=transforms.Compose([
#                            transforms.ToTensor()
#                        ])),
#         batch_size=batch_size)
    
#     print('train size: ',len(train_loader.dataset), 'test size: ',len(test_loader.dataset))
#     print('batch size:',next(iter(train_loader))[0].shape)
#     return train_loader, test_loader

import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

def data_loaders(batch_size, data_dir, seed=42, shuffle=True):
    # Set the random seed for reproducibility
    torch.manual_seed(seed)  # Set seed for PyTorch operations
    random.seed(seed)        # Set seed for Python's random module
    np.random.seed(seed)     # Set seed for numpy (which is used internally in some parts)

    # For reproducible shuffling, if shuffle is True
    if shuffle:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # In case of multiple GPUs

    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Create DataLoader for the training set
    train_loader = DataLoader(
        MNIST(root=data_dir, train=True, download=True, transform=transform),
        batch_size=batch_size,
        shuffle=shuffle
    )

    # Create DataLoader for the test set
    test_loader = DataLoader(
        MNIST(root=data_dir, train=False, transform=transform),
        batch_size=batch_size,
        shuffle=False  # Typically, you don't shuffle the test set
    )
    
    # Print dataset sizes and batch size of the first batch
    print('train size: ', len(train_loader.dataset), 'test size: ', len(test_loader.dataset))
    print('batch size:', next(iter(train_loader))[0].shape)

    return train_loader, test_loader
