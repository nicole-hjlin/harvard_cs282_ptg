# Baseline implementation

This is an implemenation of [Black 2021](https://arxiv.org/abs/2111.08230).

To do list:
- [x] Implement training of FMNIST
- - [ ] Features are normalized between 0 and 1
- [ ] Implement training of other models/datasets (optional?)
- [ ] Add random seed (RS) / Leave-One-Out (LOO) properties to State (?) class
- [ ] Train and Save Singleton Models
- - [ ] FMNIST: Train 200 models for RS, 200 models for LOO
- - [ ] Tabular: Train 500 models for RS, 500 models for LOO (optional?)
- [ ] Implement performance metrics
- - [ ] Singleton model accuracy (Table 1, RS & LOO, mean +- std. dev)
- - [ ] Flip probability (Table 2, Figure 3)
- - [ ] Selective ensemble accuracy + abstention rates (Table 3, RS & LOO)
- - [ ] Structural Similarity (SSIM) metric (Figure 3, Table 4)
- - [ ] Pearson's Correlation Coefficient (r) (Table 4)
- [ ] Evaluate ensemble against singleton models
- - [ ] Alpha is 0.05 in main paper

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
