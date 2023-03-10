# Baseline implementation

This is an implemenation of [Selective Ensembles for Consistent Predictions (Black et al. 2021)](https://arxiv.org/abs/2111.08230).

The following usage of main.py will train 200 models on FMNIST, in the same vein as the paper. Hyperparameters are set by default to those used in the paper, where listed.

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
