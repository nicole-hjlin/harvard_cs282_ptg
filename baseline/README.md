# Baseline implementation

This is an implemenation of [Selective Ensembles for Consistent Predictions (Black et al. 2021)](https://arxiv.org/abs/2111.08230).

This paper models randomness in the training process in two ways:
- The random state (RS) initialization of the network's parameters
- Leave-one-out (LOO), in which a random datapoint is ommited from the training dataset

We provide code for training on the Fashion-MNIST dataset, or FMNIST, which has 60,000 training inputs. We also randomly remove 10% of these if the --loo flag is provided, as opposed to a single datapoint, in order to gauge more strongly the effects of data removal.

**The replication of the paper's results can be found in the "experiments" notebook.**

The following usage of main.py will train 200 LeNet5 models on FMNIST, in the same vein as the paper, with default hyperparameters to match those listed in the text. Pre-trained models are also provided in the checkpoints folder.

```
$ python -m pip install -r requirements.txt
$ wandb login
$ python main.py --help
usage: main.py [-h] [--experiment EXPERIMENT] [--n N] [--a A] [--lr LR] [--loo] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--dropout DROPOUT]

options:
  -h, --help            show this help message and exit
  --experiment EXPERIMENT
                        name of experiment
  --n N                 size of ensemble
  --a A                 alpha
  --lr LR               learning rate
  --loo                 leave-one-out as source of randomness
  --epochs EPOCHS       number of epochs
  --batch_size BATCH_SIZE
                        batch size
  --dropout DROPOUT     dropout rate
```

FASRC clusters module requirements
```
module load python/3.9.12-fasrc01
module load GCC/8.2.0-2.31.1
```
