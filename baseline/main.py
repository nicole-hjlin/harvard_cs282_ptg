import util, fmnist
import argparse
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, default='glory', help='Name of experiment')
parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--dropout', type=int, default=0.1, help='Dropout rate')

wandb.init(project='ptg-baseline', config=parser.parse_args())

n = 10
g = util.sample_ensemble(
    fmnist.learning_pipeline,
    fmnist.state_sampler,
    n
)
# predict_ensemble(g, a, x)
