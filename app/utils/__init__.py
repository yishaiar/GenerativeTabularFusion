import torch
import numpy as np
import random

def set_random_seed(seed=42):
    """
    Sets the random seed for reproducibility across CPU, CUDA (if available), and MPS (Metal, if available on macOS).
    
    Args:
        seed (int): The random seed value (default is 42).
    """
    # Set Python's random seed
    random.seed(seed)
    
    # Set NumPy's random seed
    np.random.seed(seed)
    
    # Set PyTorch's random seed for CPU
    torch.manual_seed(seed)
    
    # Set the random seed for all CUDA devices, if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior on CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set the random seed for MPS (Metal backend), if available
    if torch.backends.mps.is_available():
        torch.backends.mps.deterministic = True
        torch.backends.mps.benchmark = False

    print(f"Random seed {seed} set for CPU, CUDA (if available), and MPS (if available)")

