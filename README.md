# Harvard COMPSCI 282BR
## Topics in Machine Learning: Interpretability and Explainability

Dan Ley, Matthew Nazari, Leonard Tang, Hongjin Lin

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
                 --mode_connect <CURVE> \ # default='', train models in pairs with mode connectivity (bezier or polychain)
                 --wandb \ # use to track training with weights and biases
                 --experiment <EXPERIMENT> \ # name of experiment for wandb
```

### Post-Processing Models

To compute standard/perturbed logits, predictions, explanations, first configure the postprocess_config.json file

This will specify the directory folder via name/hyperparameters, and perturbation/explanation parameters

Statistics can be computed using the following command

```
python3 postprocess.py --loo \ # use for leave-one-out as source of randomness
                       --preds \ # use to save predictions from models
                       --logits \ # use to save logits from models
                       --explanation <EXPLANATION> \ # default = '', use to save explanations by name e.g. gradients, smoothgrad, etc.
                       --mode_connect <CURVE> \ # default = '', use to load mode connected models (bezier or polychain)
                       --perturb \ # use to perturb weights before saving statistics
                       --config <CONFIG> \ # default = postprocess_config.json, use to select directory/parameters
```
