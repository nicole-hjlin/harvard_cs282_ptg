# Baseline implementation

This is an implemenation of [Black 2021](https://arxiv.org/abs/2111.08230).

To do list:
- [x] Implement training of FMNIST
- [ ] Implement training of other models/datasets (optional?)
- [ ] Evaluate performance against singleton models

```
$ python main.py --help
usage: main.py [-h] [--experiment EXPERIMENT] [--n N] [--a A] [--lr LR] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--dropout DROPOUT]

options:
  -h, --help            show this help message and exit
  --experiment EXPERIMENT
                        Name of experiment
  --n N                 Size of ensemble
  --a A                 Alpha
  --lr LR               Learning rate
  --epochs EPOCHS       Number of epochs
  --batch_size BATCH_SIZE
                        Batch size
  --dropout DROPOUT     Dropout rate
```