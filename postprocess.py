"""Postprocess models and save results"""

import json
import argparse
import torch
import numpy as np
import datasets
from datasets.tabular import TabularModelPerturb
from train import get_model_class
from style import bold
from tqdm import tqdm

def get_statistics(model_idx, method, directory):
    if method == 'average':
        grads = np.array([np.load(f'{directory}/grads_{idx}.npy') for idx in model_idx]).mean(axis=0)
        logits = np.array([np.load(f'{directory}/logits_{idx}.npy') for idx in model_idx])
        preds = logits.mean(axis=0).argmax(axis=1)
    elif method == 'majority':
        logits = np.array([np.load(f'{directory}/logits_{idx}.npy') for idx in model_idx])
        grads = np.array([np.load(f'{directory}/grads_{idx}.npy') for idx in model_idx])
        preds = np.zeros(logits.shape[1], dtype=int)
        preds[logits.argmax(2).mean(axis=0) >= 0.5] = 1
        majority_votes = (logits.argmax(axis=2) == preds)
        num_majority_votes = majority_votes.sum(axis=0)
        selected_grads = np.where(majority_votes[:, :, None], grads, 0)
        grads = selected_grads.sum(axis=0) / num_majority_votes[:, None]
    elif method == 'perturb':
        grads = np.array([np.load(f'{directory}/grads_perturb_{idx}.npy') for idx in model_idx]).mean(axis=0)
        logits = np.array([np.load(f'{directory}/logits_perturb_{idx}.npy') for idx in model_idx])
        preds = logits.mean(axis=0).argmax(axis=1)
    else:
        raise ValueError(f'Invalid method: {method}')
    return grads, preds

def load_config(config_file):
    """Load config from file"""
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def load_model(idx):
    """Load model from globals (model_class, model_args, directory)"""
    model = model_class(*model_args)
    state_dict = torch.load(f'{directory}/model_{idx}.pth')
    model.load_state_dict(state_dict)
    return model

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--loo', action='store_true', help='leave-one-out as source of randomness')
    parser.add_argument('--preds', action='store_true', help='save predictions')
    parser.add_argument('--logits', action='store_true', help='save logits')
    parser.add_argument('--explanation', type=str, default='', help='save explanations using config file parameters (gradient, smoothgrad, etc.)')
    parser.add_argument('--perturb', action='store_true', help='perturb weights and save mean results')
    parser.add_argument('--config', type=str, default='postprocess_config.json', help='config file to load')

    # Get config dictionary
    args = vars(parser.parse_args())
    config = load_config(args['config'])
    exp = args['explanation']
    if exp != '':
        exp_params = config['explanations'][exp]
        print(bold("Explanation:"), exp)
        print(bold("Explanation parameters:"), exp_params)
    
    # Load dataset
    name = config['name']
    trainset, testset = datasets.load_dataset(name)
    X_test, y_test = testset.data.numpy(), testset.labels.numpy()
    n_inputs, n_features = X_test.shape

    # Determine directory
    random_source = 'loo' if args['loo'] else 'rs'
    hyperparameters = config['hyperparameters']
    optim = hyperparameters['optimizer']
    epochs = hyperparameters['epochs']
    lr = hyperparameters['learning_rate']
    batch_size = hyperparameters['batch_size']
    dropout = hyperparameters['dropout']
    directory = f'models/{name}/{random_source}/{optim}_epochs{epochs}_lr{lr}_batch{batch_size}_dropout{dropout}'
    print(bold("Directory:"), directory)

    # Save config file to directory
    with open(f'{directory}/postprocess_config.json', 'w') as f:
        json.dump(config, f)

    # Determine model class and arguments
    model_class = get_model_class(name)
    if name == 'fmnist':
        model_args = [10, dropout]
    elif name in ['german', 'adult', 'heloc']:
        model_args = [n_features, datasets.tabular.layers[name]]

    log = 'logits' if args['logits'] else ''
    pred = 'preds' if args['preds'] else ''
    exp = args['explanation']

    if (log == '') and (pred == '') and (exp == ''):
        print(bold("Nothing to do! Specify --logits, --preds, or --explanation."))
    else:
        # Join log, pred and explanation into one string
        statistics = [s for s in [log, pred, exp] if s]
        stat_str = ', '.join(statistics)
        print(bold(f"Computing statistics for {stat_str}"))
        if args['perturb']:
            print(bold("Perturbing weights"))

        # Compute statistics
        for i in tqdm(range(config['n'])):
            # Load model
            model = load_model(i)
            if args['perturb']:
                n_weight_perturbations = config['perturb']['n_weight_perturbations']
                weight_sigma = config['perturb']['weight_sigma']
                model = TabularModelPerturb(model, n_weight_perturbations, weight_sigma)  # No FMNIST yet

            # Compute predictions
            if pred:
                if args['perturb']:
                    preds = model.predict(X_test, mean=True)
                    perturb = 'perturb_mean_'
                else:
                    preds = model.predict(X_test, return_numpy=True)
                    perturb = ''
                np.save(f'{directory}/preds_{perturb}{i}.npy', preds)

            # Compute logits
            if log:
                if args['perturb']:
                    logits = model.forward(torch.FloatTensor(X_test)).detach().numpy().mean(axis=0)
                    perturb = 'perturb_'
                else:
                    logits = model.forward(torch.FloatTensor(X_test)).detach().numpy()
                    perturb = ''
                np.save(f'{directory}/logits_{perturb}{i}.npy', logits)

            # Compute explanations
            if exp == 'gradient':
                if args['perturb']:
                    grads = model.compute_gradients(X_test, mean=True)
                    perturb = 'perturb_'
                else:
                    grads = model.compute_gradients(X_test, return_numpy=True)
                    perturb = ''
                np.save(f'{directory}/grads_{perturb}{i}.npy', grads)
            else:
                pass  # TODO: implement other explanations
