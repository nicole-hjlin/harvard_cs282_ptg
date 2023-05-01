"""Utility functions for the baseline
including state samplers, learning pipelines, and ensembling methods"""

from typing import Callable
from dataclasses import dataclass
import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np

@dataclass
class State:
    """State object for the learning pipeline"""
    net: nn.Module
    trainset: Dataset
    testset: Dataset
    seed: int
    config: dict

# Type aliases
LearningPipeline = Callable[[State], State]

def get_optimizer(S):
    if S.config['optimizer'].lower() == 'adam':
        optimizer = torch.optim.Adam(
            S.net.parameters(),
            lr=S.config['lr'],
        )
    elif S.config['optimizer'].lower() == 'sgd':
        optimizer = torch.optim.SGD(
            S.net.parameters(),
            lr=S.config['lr'],
        )
    else:
        raise ValueError(f"Optimizer {S.config['optimizer']} not recognized")
    return optimizer

def convert_to_tensor(arr):
    """Conditional conversion to a torch tensor"""

    # Return torch tensor
    return torch.from_numpy(arr) if isinstance(arr, np.ndarray) else arr

def convert_to_numpy(arr):
    """Conditional conversion to a numpy array"""

    # Return numpy array
    return arr.numpy() if isinstance(arr, torch.Tensor) else arr