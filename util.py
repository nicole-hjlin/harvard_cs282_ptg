"""Utility functions for the baseline
including state samplers, learning pipelines, and ensembling methods"""

import os
from typing import Callable, List, Union
from dataclasses import dataclass
import torch
from torch import nn, optim
from torch.utils.data import Dataset
import numpy as np
import wandb
from scipy.stats import binom

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

def get_optimizer(S: State):
    if S.config['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(
            S.net.parameters(),
            lr=S.config['lr'],
        )
    elif S.config['optimizer'].lower() == 'sgd':
        optimizer = optim.SGD(
            S.net.parameters(),
            lr=S.config['lr'],
        )
    else:
        raise ValueError(f"Optimizer {S.config['optimizer']} not recognized")
    return optimizer

def train_ensemble(
    P: LearningPipeline,
    S: List[State],
    n: int,
) -> List[nn.Module]:
    """Trains an ensemble of n models given a learning pipeline P and a list of states S"""

    # Initialize ensemble
    g = []
    config = wandb.config

    # Train each model
    for i in range(n):
        # Start wandb run
        wandb.init(
            project='ptg-baseline',
            group=config['experiment'],
            name=f'model{i}',
            config=config,
        )

        # Train model
        model = P(S[i])
        g.append(model)

        # Save model
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pth'))

        # Finish wandb run
        wandb.finish()

    # Return the ensemble
    return g


# def sample_ensemble(
#     P: LearningPipeline,
#     S: StateSampler,
#     n: int,
# ) -> list[nn.Module]:
#     """Samples an ensemble of n models given a
#     learning pipeline P and a state sampler S
#     Returns: list of trained models"""

#     # Sample states
#     S = [S() for _ in range(n)]

#     # Return list of trained models
#     return train_ensemble(P, S, n)

def combine_preds(directory, random_source, idx):
    """Get predictions from a directory, given a random source and indices
    Returns the predictions in a numpy array"""
    preds = []
    for i in idx:
        preds.append(np.load(f'{directory}/{random_source}_preds_{i}.npy'))
    return np.array(preds)


def convert_to_numpy(
    arr: Union[torch.Tensor, np.ndarray],
) -> np.ndarray:
    """Conditional conversion to a numpy array"""

    # Return numpy array
    return arr.numpy() if isinstance(arr, torch.Tensor) else arr


def convert_to_tensor(
    arr: Union[torch.Tensor, np.ndarray],
) -> np.ndarray:
    """Conditional conversion to a torch tensor"""

    # Return torch tensor
    return torch.from_numpy(arr) if isinstance(arr, np.ndarray) else arr


def select_preds(
    preds: Union[torch.Tensor, np.ndarray],
    a: float = 0.05,
    ensemble_method: str = 'selective',
) -> np.ndarray:
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
                # Compute two tailed binomial test
                # assuming equal chance between c_a and c_b
                p = 2*(1-binom.cdf(n_a-1, n_a+n_b, 0.5))
                # built in function is slower:
                # from scipy.stats import binomtest
                # p = binomtest(n_a, n_a+n_b, 0.5,
                    #           alternative='two-sided').pvalue
                if p < a:
                    ensemble_preds[i] = c_a
                else:
                    # np.inf since np.nan cannot be compared easily
                    ensemble_preds[i] = np.inf

        # Handle unknown ensemble method
        else:
            raise ValueError('Unknown Ensembling Method')

    # Return ensemble predictions
    return ensemble_preds


def compute_top2(
    preds: np.ndarray,
):
    """Takes an array of input-specific predictions of size (no. models)
    Returns the top two classes (c_a, c_b) and their frequencies (n_a, n_b)"""

    # Count class occurrences
    counts = torch.bincount(preds)

    if len(counts) == 1: # handle edge cases
        c_a, c_b, n_a, n_b = 0, None, counts.item(), 0
    else:  # compute top two classes and their frequencies
        n, c = counts.topk(2)  # e.g. c = [5, 3] and n = [12, 8]
        c_a, c_b = c.tolist()  # following Appendix notation
        n_a, n_b = n.tolist()
        c_b = None if n_b == 0 else c_b

    # Return top two classes and their frequencies
    return c_a, c_b, n_a, n_b
