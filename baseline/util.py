from typing import Callable
import torch
from torch import nn
from torch.utils.data import Dataset
from dataclasses import dataclass
import wandb

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
        g.append(P(S[i]))
        wandb.finish()
    # g = [P(S[i]) for i in range(n)]
    return g

def sample_ensemble(
    P: LearningPipeline,
    S: StateSampler,
    n: int,
):
    S = [S() for _ in range(n)]
    return train_ensemble(P, S, n)

def predict_ensemble(
    g: list[nn.Module],
    a: float,
    x: torch.Tensor,
):
    y_pred = sum(net.eval()(x) for net in g)
    p = -1 # idk what the paper means here
    if p < a:
        return y_pred.argmax(-1)
    else:
        return None