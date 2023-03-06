import util, fmnist
import argparse
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, default='glory', help='Name of experiment')
parser.add_argument('--n', type=int, default=500, help='Size of ensemble')
parser.add_argument('--a', type=float, default=0.05, help='Alpha')
parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--dropout', type=int, default=0.1, help='Dropout rate')
config = vars(parser.parse_args())
wandb.config = config

n = config['n']
g = util.sample_ensemble(
    fmnist.learning_pipeline,
    fmnist.state_sampler,
    n,
)

a = config['a']
x = fmnist.trainset[0][0].unsqueeze(0)
pred = util.predict_ensemble(g, a, x)
