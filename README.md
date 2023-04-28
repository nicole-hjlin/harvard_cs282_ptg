# Harvard COMPSCI 282BR
## Topics in Machine Learning: Interpretability and Explainability

Dan Ley, Hongjin Lin, Leonard Tang, Matthew Nazari

2023 Spring

## Usage

### Training Models

Models can be trained using the following command

```
python3 train.py --name <DATASET NAME> \ # default = heloc
                 --n <NO. MODELS> \ # default = 20
                 --loo \ # use for leave-one-out as source of randomness
                 --lr <LEARNING RATE> \ # default = 0.1
                 --epochs <EPOCHS> \ # default = 20
                 --batch_size <BATCH_SIZE> \ # default = 64
                 --dropout <DROPOUT> \ # default = 0
                 --optimizer <OPTIMIZER> \ # default = sgd
                 --wandb \ # use to track training with weights and biases
                 --experiment <EXPERIMENT> \ # name of experiment for wandb
```