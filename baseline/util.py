from typing import Callable
import torch
from torch import nn
from torch.utils.data import Dataset
from scipy.stats import binom
from dataclasses import dataclass
import numpy as np
import wandb
import os

@dataclass
class State:
    net: nn.Module
    trainset: Dataset
    hyperparameters: dict

StateSampler = Callable[[], State]

LearningPipeline = Callable[[State], State]

def train_ensemble(
    P: LearningPipeline,
    S: list[State],
    n: int,
):
    g = []
    config = wandb.config
    for i in range(n):
        wandb.init(
            project='ptg-baseline',
            group=config['experiment'],
            name=f'model{i}',
            config=config,
        )
        model = P(S[i])
        g.append(model)
        # save model
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pth'))

        wandb.finish()
    return g

def sample_ensemble(
    P: LearningPipeline,
    S: StateSampler,
    n: int,
):
    S = [S() for _ in range(n)]
    return train_ensemble(P, S, n)

def select_preds(
    preds: torch.Tensor | np.ndarray,
    a: float = 0.05,
    ensemble_method: str = 'selective',
):
    """Takes an array of predictions of size (no. models, no.inputs)
    Returns a numpy prediction array of size (no.inputs)
    A prediction of np.inf represents abstention"""

    # Convert to tensor
    preds = convert_to_tensor(preds)

    # Initialize ensemble predictions
    n_inputs = preds.shape[1]
    ensemble_preds = np.zeros(n_inputs)

    # For each input
    for i in range(n_inputs):
        # Compute top two classes and their frequencies
        c_a, _, n_a, n_b = compute_top2(preds[:,i])

        # Simple majority vote
        if ensemble_method == 'majority':
            ensemble_preds[i] = c_a

        # Selective majority vote
        elif ensemble_method == 'selective':
            # If all models predicted c_a
            if n_b == 0:
                ensemble_preds[i] = c_a
            else:
                # Compute binomial probability of n_a
                # assuming equal chance between c_a and c_b
                p = binom.pmf(n_a, n_a+n_b, 0.5)
                if p < a:
                    ensemble_preds[i] = c_a
                else:
                    # np.inf since np.nan cannot be compared easily
                    ensemble_preds[i] = np.inf

        else:
            raise ValueError('Unknown Ensembling Method')

    return ensemble_preds

def compute_top2(
    preds: np.ndarray,
):
    # Compute top 2 classes (c) and their frequencies (n)
    # For a given input, across multiple model predictions
    # Size of preds is no. models
    # E.g. c = [5, 3] and n = [22, 20]
    # (classes 5 and 3 recieved 20 and 22 votes)

    # takes preds for one input
    counts = torch.bincount(preds)
    if len(counts) == 1:  # handles case where all predictions are 0
        return 0, None, counts.item(), 0
    n, c = counts.topk(2)
    c_a, c_b = c.tolist() # following Appendix notation
    n_a, n_b = n.tolist()
    c_b = None if n_b == 0 else c_b
    return c_a, c_b, n_a, n_b

def convert_to_numpy(
    arr: torch.Tensor | np.ndarray,
):
    # Conditional conversion that has widespread use
    return arr.numpy() if isinstance(arr, torch.Tensor) else arr

def convert_to_tensor(
    arr: torch.Tensor | np.ndarray,
):
    # Conditional conversion that has widespread use
    return torch.from_numpy(arr) if isinstance(arr, np.ndarray) else arr

def get_folder_names(directory):
    folder_names = {'rs': [], 'loo': []}
    for entry in os.scandir(directory):
        if entry.is_dir():
            key = entry.name.split('_')[0]
            folder_names[key].append(entry.name)
    return folder_names

# def equal_arrays_nan(
#     a: np.ndarray,
#     b: np.ndarray,
# ):
#     # functionality for np.nan
#     return ((a == b) | (np.isnan(a) & np.isnan(b)))