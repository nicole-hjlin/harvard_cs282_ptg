# Baseline implementation

This is an implemenation of [Black 2021](https://arxiv.org/abs/2111.08230).

To do list:
- [x] Implement training of FMNIST
- [ ] Implement training of other models/datasets (optional?)
- [ ] Implement performance metrics
- [ ] Evaluate ensemble against singleton models


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