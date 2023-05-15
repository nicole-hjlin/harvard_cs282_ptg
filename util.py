"""Utility functions for the baseline
including state samplers, learning pipelines, and ensembling methods"""

from typing import Callable
from dataclasses import dataclass
import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np

_exp_str_dict = {'gradients': 'grads',
                 'smoothgrad': 'sg',
                 'shap': 'shaps'}

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

def get_statistics(model_idx, method, directory, exp='gradients'):
    exp_str = _exp_str_dict[exp]
    if method == 'average':
        grads = np.array([np.load(f'{directory}/{exp_str}_{idx}.npy') for idx in model_idx]).mean(axis=0)
        logits = np.array([np.load(f'{directory}/logits_{idx}.npy') for idx in model_idx])
        preds = logits.mean(axis=0).argmax(axis=1)
    elif method == 'majority':
        logits = np.array([np.load(f'{directory}/logits_{idx}.npy') for idx in model_idx])
        grads = np.array([np.load(f'{directory}/{exp_str}_{idx}.npy') for idx in model_idx])
        preds = np.zeros(logits.shape[1], dtype=int)
        preds[logits.argmax(2).mean(axis=0) >= 0.5] = 1

        preds_trunc = preds[:grads.shape[1]]
        logits_trunc = logits[:, :grads.shape[1]]
        majority_votes = (logits_trunc.argmax(axis=2) == preds_trunc)
        num_majority_votes = majority_votes.sum(axis=0)
        selected_grads = np.where(majority_votes[:, :, None], grads, 0)
        
        grads = selected_grads.sum(axis=0) / num_majority_votes[:, None]
    elif method == 'perturb':
        grads = np.array([np.load(f'{directory}/{exp_str}_perturb_{idx}.npy') for idx in model_idx]).mean(axis=0)
        logits = np.array([np.load(f'{directory}/logits_perturb_{idx}.npy') for idx in model_idx])
        preds = logits.mean(axis=0).argmax(axis=1)
    elif method == 'mode connect':
        # Take half of the models
        model_idx = model_idx[:len(model_idx)//2]
        grads = np.array([np.load(f'{directory}/{exp_str}_bezier_{idx}.npy') for idx in model_idx]).mean(axis=0)
        logits = np.array([np.load(f'{directory}/logits_bezier_{idx}.npy') for idx in model_idx])
        preds = logits.mean(axis=0).argmax(axis=1)
    elif method == 'combined':
        model_idx = model_idx[:len(model_idx)//2]
        grads = np.array([np.load(f'{directory}/{exp_str}_bezier_perturb_{idx}.npy') for idx in model_idx]).mean(axis=0)
        logits = np.array([np.load(f'{directory}/logits_bezier_perturb_{idx}.npy') for idx in model_idx])
        preds = logits.mean(axis=0).argmax(axis=1)
    else:
        raise ValueError(f'Invalid method: {method}')
    return grads, preds

def get_weight_diff(state_dict1, state_dict2):
    if isinstance(state_dict1, nn.Module):
        state_dict1 = state_dict1.state_dict()
    if isinstance(state_dict2, nn.Module):
        state_dict2 = state_dict2.state_dict()
    diff = 0
    for k in state_dict1.keys():
        diff += np.linalg.norm(state_dict1[k] - state_dict2[k])**2
    return diff**0.5

def get_weight_norm(state_dict):
    if isinstance(state_dict, nn.Module):
        state_dict = state_dict.state_dict()
    norm = 0
    for k in state_dict.keys():
        norm += np.linalg.norm(state_dict[k])**2
    return norm**0.5

def linear_weight_interpolation(state_dict1, state_dict2, ts):
    if isinstance(state_dict1, nn.Module):
        state_dict1 = state_dict1.state_dict()
    if isinstance(state_dict2, nn.Module):
        state_dict2 = state_dict2.state_dict()
    # Interpolate between two state dicts
    state_dicts = []
    for t in ts:
        state_dict = {}
        for key in state_dict1.keys():
            state_dict[key] = state_dict1[key] + (state_dict2[key] - state_dict1[key]) * t
        state_dicts.append(state_dict)
    return state_dicts
