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

def get_statistics(model_idx, method, directory):
    if method == 'average':
        grads = np.array([np.load(f'{directory}/grads_{idx}.npy') for idx in model_idx]).mean(axis=0)
        logits = np.array([np.load(f'{directory}/logits_{idx}.npy') for idx in model_idx])
        preds = logits.mean(axis=0).argmax(axis=1)
    elif method == 'majority':
        logits = np.array([np.load(f'{directory}/logits_{idx}.npy') for idx in model_idx])
        grads = np.array([np.load(f'{directory}/grads_{idx}.npy') for idx in model_idx])
        preds = np.zeros(logits.shape[1], dtype=int)
        preds[logits.argmax(2).mean(axis=0) >= 0.5] = 1
        majority_votes = (logits.argmax(axis=2) == preds)
        num_majority_votes = majority_votes.sum(axis=0)
        selected_grads = np.where(majority_votes[:, :, None], grads, 0)
        grads = selected_grads.sum(axis=0) / num_majority_votes[:, None]
    elif method == 'perturb':
        grads = np.array([np.load(f'{directory}/grads_perturb_{idx}.npy') for idx in model_idx]).mean(axis=0)
        logits = np.array([np.load(f'{directory}/logits_perturb_{idx}.npy') for idx in model_idx])
        preds = logits.mean(axis=0).argmax(axis=1)
    else:
        raise ValueError(f'Invalid method: {method}')
    return grads, preds