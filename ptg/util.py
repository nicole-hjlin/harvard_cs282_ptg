"""Utility functions for the baseline
including state samplers, learning pipelines, and ensembling methods"""

import os
from typing import Callable
from dataclasses import dataclass
import torch
from torch import nn
from torch.utils.data import Dataset
import wandb

@dataclass
class State:
    """State object for the learning pipeline"""
    net: nn.Module
    trainset: Dataset
    hyperparameters: dict

# Type aliases
StateSampler = Callable[[], State]
LearningPipeline = Callable[[State], State]

def train_ensemble(
    P: LearningPipeline,
    S: list[State],
    n: int,
) -> list[nn.Module]:
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


def sample_ensemble(
    P: LearningPipeline,
    S: StateSampler,
    n: int,
) -> list[nn.Module]:
    """Samples an ensemble of n models given a
    learning pipeline P and a state sampler S"""

    # Sample states
    S = [S() for _ in range(n)]

    # Return list of trained models
    return train_ensemble(P, S, n)
