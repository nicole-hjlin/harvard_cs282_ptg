import argparse
import torch
import util, fmnist
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, default='glory', help='name of experiment')
parser.add_argument('--n', type=int, default=200, help='size of ensemble')
parser.add_argument('--a', type=float, default=0.05, help='alpha')
parser.add_argument('--lr', type=float, default=1e-1, help='learning rate')
parser.add_argument('--loo', action='store_true', help='leave-one-out as source of randomness')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--dropout', type=int, default=0.05, help='dropout rate')
config = vars(parser.parse_args())
wandb.config = config

n = config['n']
g = util.sample_ensemble(
    fmnist.learning_pipeline,
    fmnist.state_sampler,
    n,
)

for i, model in enumerate(g):
    prefix = 'loo_' if config['loo'] else 'rs_'
    torch.save(model.state_dict(), f'checkpoints/{prefix}model{i}.pt')
