"""Main script for training models and saving weights/predictions"""

# Standard library imports
import argparse
import train
import datasets
import wandb
# from torch.utils.data import DataLoader

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, default='glory', help='name of experiment')
parser.add_argument('--name', type=str, default='fmnist', help='Name of the dataset (fmnist, german, heloc, etc.)')
parser.add_argument('--n', type=int, default=20, help='number of models to train')
parser.add_argument('--a', type=float, default=0.05, help='alpha')
parser.add_argument('--loo', action='store_true', help='leave-one-out as source of randomness')
parser.add_argument('--lr', type=float, default=1e-1, help='learning rate')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--dropout', type=float, default=0, help='dropout rate')
parser.add_argument('--optimizer', type=str, default='adam', help='optimizer (adam or sgd)')
parser.add_argument('--wandb', action='store_true', help='use weights and biases monitoring')


config = vars(parser.parse_args())
use_wandb = config['wandb']
if use_wandb:
    wandb.config = config
print(config)

trainset, testset = datasets.load_dataset(config['name'])
# train_dataloader = DataLoader(trainset, batch_size=config['batch_size'], shuffle=False)
# test_dataloader = DataLoader(testset, batch_size=config['batch_size'], shuffle=False)

# print(len(trainset))
# print(trainset[0][0].shape)

# Run experiment

# Number of models to train
n = config['n']

# Get learning pipeline from dataset name
learning_pipeline = train.get_learning_pipeline(config['name'])

# Get model class from dataset name
model_class = train.get_model_class(config['name'])

# Get list of n states from model class, trainset and config
# put everything in config and just pass that?
states = train.get_states(n, model_class, trainset, testset, config)

# Train models (one per state in states)
train.train_models(
    learning_pipeline,
    states,
    config,
)

# While the default is to use shell scripts to run on the cluster,
# the following code can be uncommented when training/saving models locally
# Note that the models will also still be saved on wandb
# This code also currently saves test predictions in the checkpoints folder
# 10,0000 predictions x 200 models x 4 bytes/input ~ 8MB

# import torch
# import numpy as np
# from fmnist import testset
# from torch.utils.data import DataLoader
# for i, model in enumerate(g):
#     prefix = 'loo_' if config['loo'] else 'rs_'
#     torch.save(model.state_dict(), f'checkpoints/{prefix}model{i}.pth')
#     # ~40 seconds to compute 10,000 predictions x 200 models
#     test = DataLoader(testset, batch_size=len(testset), shuffle=False)
#     X_test = next(iter(test))[0]
#     preds = compute_preds(g, X_test, return_numpy=True)
#     np.save(f'checkpoints/{prefix}_preds_fmnist_{n}.npy', preds)
