from typing import Callable
import torch
from torch import nn
from torch.utils.data import Dataset
from scipy.stats import binom
from dataclasses import dataclass
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

def predict_ensemble(
    g: list[nn.Module],
    a: float,
    x: torch.Tensor,
):
    # Assuming model(x) returns softmax vector of length no. classes
    softmax_preds = torch.stack([model(x)[0] for model in g])
    counts = torch.bincount(softmax_preds.argmax(1))

    # Compute top 2 classes (c) and their frequencies (n)
    # E.g. c = [5, 3] and n = [22, 20]
    # (classes 5 and 3 recieved 20 and 22 votes)
    n, c = counts.topk(2)
    n_a, n_b = n.tolist()
    c_a, c_b = c.tolist()  # following Appendix notation

    # Compute binomial probability of n_a,
    # assuming equal chance between c_a and c_b
    p = binom.pmf(n_a, n_a+n_b, 0.5)
    if p < a:
        return c_a
    else:
        return None