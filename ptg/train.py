import wandb
import torch
from torch import nn
from torch.utils.data import Dataset, Subset
from os import makedirs, path
from util import State, LearningPipeline
import json
import datasets
from datasets.tabular import TabularSubset

def get_model_class(name: str) -> nn.Module:
    """Returns a model class by dataset name"""
    if name == 'fmnist':
        model_class = datasets.fmnist.LeNet5
    elif name in ['german', 'adult', 'heloc']:
        model_class = datasets.tabular.TabularModel
    else:
        raise ValueError(f'Unknown dataset name: {name}')
    return model_class

def get_learning_pipeline(name: str) -> LearningPipeline:
    """Returns a learning pipeline by dataset name"""
    if name == 'fmnist':
        learning_pipeline = datasets.fmnist.learning_pipeline
    elif name in ['german', 'adult', 'heloc']:
        learning_pipeline = datasets.tabular.learning_pipeline
    else:
        raise ValueError(f'Unknown dataset name: {name}')
    return learning_pipeline
    
def get_states(
    n: int,
    model: nn.Module,
    trainset: Dataset,
    testset: Dataset,
    config: dict,
) -> list[State]:
    """Returns a list of states for the learning pipeline

    Warning: trainset needs to point to (a subset of) the original dataset,
    not a copy of it, because the trainset is permuted in the
    learning pipeline if loo is True (don't want to store all copies).

    Warning: net should also point to the original model class,
    not a copy of it, because the model is instantiated in the
    learning pipeline (don't want to store all copies).
    
    We also want to set the seed in the learning pipeline, not here or when
    we construct the states, because we want to set the seed for each model.
    """
    States = []
    for i in range(n):
        # Permute trainset if loo
        if config['loo']:
            # Permute trainset with seed
            torch.manual_seed(i)
            mask = torch.randperm(len(trainset))
            subset = TabularSubset(trainset, mask[:int(0.9 * len(mask))])
        else:
            subset = trainset


        # Create state
        S = State(
            net=model,
            trainset=subset,
            testset=testset,
            seed=i,
            config=config,
        )

        # Append state
        States.append(S)

    return States

def train_models(
    P: LearningPipeline,
    States: list[State],
    config: dict,
) -> list[nn.Module]:
    """Trains n models given a learning pipeline P and a list of states S"""

    # Initialize variables
    r = 'loo' if config['loo'] else 'rs'
    name = config['name']
    directory = f'models/{name}'

    # Save config dictionary
    if not path.exists(directory):
        makedirs(directory)
    with open(f'{directory}/{r}_config.json', 'w') as f:
        json.dump(config, f)

    # Train each model
    for i, S in enumerate(States):
        # Start wandb run
        if config['wandb']:
            wandb.init(
                project='ptg-baseline',
                group=config['experiment'],
                name=f'model{i}',
                config=config,
            )

        # Train model
        model = P(S)

        # Save model
        torch.save(model.state_dict(), f'{directory}/{r}_model_{i}.pth')

        # Finish wandb run
        if config['wandb']:
            wandb.finish()