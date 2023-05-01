"""Train model script and helper functions"""

import json
import argparse
from os import makedirs, path
import torch
from torch import nn
from torch.utils.data import Dataset
import wandb
from util import State, LearningPipeline
import datasets
from datasets.tabular import TabularSubset
from typing import List

def get_states(
    n_states: int,
    model: nn.Module,
    trainset: Dataset,
    testset: Dataset,
    config: dict,
) -> List[State]:
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
    for i in range(n_states):
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
    States: List[State],
    config: dict,
) -> List[nn.Module]:
    """Trains n models given a learning pipeline P and a list of states S"""

    # Initialize variables
    r = 'loo' if config['loo'] else 'rs'
    name = config['name']
    optim = config['optimizer']
    epochs = config['epochs']
    lr = config['lr']
    batch_size = config['batch_size']
    dropout = config['dropout']
    directory = f'models/{name}/{r}/{optim}_epochs{epochs}_lr{lr}_batch{batch_size}_dropout{dropout}'

    # Save config dictionary
    if not path.exists(directory):
        makedirs(directory)
    with open(f'{directory}/config.json', 'w') as f:
        json.dump(config, f)

    # Train each model
    for i, S in enumerate(States):
        # Start wandb run
        if config['wandb']:
            wandb.init(
                project='baseline',
                group=config['experiment'],
                name=f'model{i}',
                config=config,
            )

        # Train model
        model = P(S)

        # Save model
        mname = 'model' if config['mode_connect']=='' else config['mode_connect']
        torch.save(model.state_dict(), f'{directory}/{mname}_{i}.pth')

        # Finish wandb run
        if config['wandb']:
            wandb.finish()


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='heloc',
                        help='Name of the dataset (fmnist, german, heloc, etc.)')
    parser.add_argument('--n', type=int, default=20, help='number of models to train')
    parser.add_argument('--loo', action='store_true', help='leave-one-out as source of randomness')
    parser.add_argument('--lr', type=float, default=1e-1, help='learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--dropout', type=float, default=0, help='dropout rate')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer (adam or sgd)')
    parser.add_argument('--mode_connect', type=str, default='', help='train models in pairs with mode connectivity (bezier or polychain)')
    parser.add_argument('--wandb', action='store_true', help='use weights and biases monitoring')
    parser.add_argument('--experiment', type=str, default='training', help='name of experiment for wandb')

    # Get config dictionary
    config = vars(parser.parse_args())
    use_wandb = config['wandb']
    if use_wandb:
        wandb.config = config
    print(config)

    # Load dataset
    trainset, testset = datasets.load_dataset(config['name'])

    # Number of models to train
    n = config['n']

    # Get model class from dataset name
    model_class = datasets.get_model_class(config['name'])

    # Get learning pipeline from dataset name
    learning_pipeline = datasets.get_learning_pipeline(config['name'])

    # Get list of n states from model class, trainset and config
    states = get_states(n, model_class, trainset, testset, config)

    # Train models (one per state in states)
    train_models(
        learning_pipeline,
        states,
        config,
    )
