"""Main script for training models and saving predictions"""

# Standard library imports
import argparse
import util
import datasets
import wandb
from torch.utils.data import DataLoader

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, default='glory', help='name of experiment')
parser.add_argument('--name', type=str, default='fmnist', help='Name of the dataset (fmnist, german, heloc, etc.)')
parser.add_argument('--n', type=int, default=200, help='size of ensemble')
parser.add_argument('--a', type=float, default=0.05, help='alpha')
parser.add_argument('--lr', type=float, default=1e-1, help='learning rate')
parser.add_argument('--loo', action='store_true', help='leave-one-out as source of randomness')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--dropout', type=int, default=0.05, help='dropout rate')


config = vars(parser.parse_args())
wandb.config = config

trainset, testset = datasets.load_dataset(config['name'])
train_dataloader = DataLoader(trainset, batch_size=config['batch_size'], shuffle=False)
test_dataloader = DataLoader(testset, batch_size=config['batch_size'], shuffle=False)
print(len(trainset))
print(trainset[0])

# print(len(fmnist.dataset))
# print(len(fmnist.dataset[0]))
# print(fmnist.dataset[0])

# # Run experiment
# n = config['n']
# g = util.sample_ensemble(
#     german.learning_pipeline,
#     german.state_sampler,
#     n,
# )

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
