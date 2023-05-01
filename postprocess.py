"""Postprocess models and save results"""

import json
import argparse
import torch
import numpy as np
import datasets
from modconn import curves
from datasets.tabular import TabularModelPerturb
from datasets import get_model_class, get_curve_class
from style import bold
from tqdm import tqdm

_curve_dict = {'bezier': curves.Bezier, 'polychain': curves.PolyChain}

def _load_config(config_file):
    """Load config from file"""
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def _load_model(idx):
    """Load model from globals (model_class, model_args, directory)"""
    # Load perturbed model (implement perturbations for mode connectivity)
    if mode_connect:
        model = curves.CurveNet(*curve_args)
        state_dict = torch.load(f'{directory}/{curve_type}_{idx}.pth')
        model.load_state_dict(state_dict)
    else:
        model = model_class(*model_args)
        state_dict = torch.load(f'{directory}/model_{idx}.pth')
        model.load_state_dict(state_dict)
        if perturb:
            model = TabularModelPerturb(model, n_weight_perturbations, weight_sigma)  # No FMNIST yet
    return model

def _get_logits():
    """Get logits from globals (model, model_class, X_test, mode_connect, perturb)"""
    if mode_connect:
        if perturb:
            pass
        else:
            logits = model.compute_logits(X_test, model_class, ts).mean(axis=0)
    else:
        if perturb:
            logits = model.forward(torch.FloatTensor(X_test)).detach().numpy().mean(axis=0)
        else:
            logits = model.forward(torch.FloatTensor(X_test)).detach().numpy()
    return logits

def _get_grads():
    if mode_connect:
        if perturb:
            pass
        else:
            grads = model.compute_gradients(X_test, model_class, ts).mean(axis=0)
    else:
        if perturb:
            grads = model.compute_gradients(X_test, mean=True)
        else:
            grads = model.compute_gradients(X_test, return_numpy=True)
    return grads

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--loo', action='store_true', help='leave-one-out as source of randomness')
    parser.add_argument('--preds', action='store_true', help='save predictions')
    parser.add_argument('--logits', action='store_true', help='save logits')
    parser.add_argument('--explanation', type=str, default='', help='save explanations using config file parameters (gradient, smoothgrad, etc.)')
    parser.add_argument('--mode_connect', action='store_true', help='load mode connected models')
    parser.add_argument('--perturb', action='store_true', help='perturb weights and save mean results')
    parser.add_argument('--config', type=str, default='postprocess_config.json', help='config file to load')

    # Get config dictionary
    args = vars(parser.parse_args())
    config = _load_config(args['config'])
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

    # Perturbation
    perturb = 'perturb_' if args['perturb'] else ''
    if perturb:
        n_weight_perturbations = config['perturb']['n_weight_perturbations']
        weight_sigma = config['perturb']['weight_sigma']

    # Mode connectivity
    mode_connect = ''
    if args['mode_connect']:
        curve_type = config['mode_connect']['curve_type']
        mode_connect = curve_type + '_'
        n_curve_samples = config['mode_connect']['n_curve_samples']
        ts = np.linspace(0, 1, n_curve_samples)
        curve_class = get_curve_class(name)
        curve_args = [_curve_dict[curve_type], curve_class, 2, n_features,
                      datasets.tabular.layers[name], False, False]

    # Determine statistics to compute
    log = 'logits' if args['logits'] else ''
    pred = 'preds' if args['preds'] else ''
    exp = args['explanation']

    # Compute statistics
    if not (log or pred or exp):
        # Nothing to do
        print(bold("Nothing to do! Specify --logits, --preds, or --explanation."))
    else:
        # Which statistics?
        statistics = [s for s in [log, pred, exp] if s]
        stat_str = ', '.join(statistics)
        print(bold(f"Computing statistics for {stat_str}"))

        # Perturbation/Mode connectivity
        if perturb:
            print(bold("Perturbing weights"))
        if mode_connect:
            print(bold("Mode connectivity"))
            print(bold("Curve type:"), curve_type)
            print(bold("Number of curve samples:"), n_curve_samples)

        # Compute statistics
        for i in tqdm(range(config['n'])):

            # Load model
            model = _load_model(i)

            # Compute logits
            if log or pred:
                logits = _get_logits()
                if log:
                    np.save(f'{directory}/logits_{mode_connect}{perturb}{i}.npy', logits)
                if pred:
                    np.save(f'{directory}/preds_{mode_connect}{perturb}{i}.npy', np.argmax(logits, axis=1))

            # Compute explanations
            if exp == 'gradient':
                grads = _get_grads()
                np.save(f'{directory}/grads_{mode_connect}{perturb}{i}.npy', grads)
            else:
                pass  # TODO: implement other explanations
