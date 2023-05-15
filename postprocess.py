"""Postprocess models and save results"""

import json
import argparse
import torch
import numpy as np
import datasets
import curves
from datasets.tabular import TabularModel, TabularModelPerturb
from datasets import get_model_class, get_curve_class
from style import bold
from multiprocessing import set_start_method, cpu_count
from joblib import Parallel, delayed
import joblib
from tqdm import tqdm
import contextlib
import time

_curve_dict = {'bezier': curves.Bezier, 'polychain': curves.PolyChain}

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

def _get_model_stats(i):
    # Load model
    model = _load_model(i)

    # Compute logits
    if log or pred:
        logits = _get_logits(model)
        if log:
            np.save(f'{directory}/logits_{mode_connect}{perturb}{i}.npy', logits)
        if pred:
            np.save(f'{directory}/preds_{mode_connect}{perturb}{i}.npy', np.argmax(logits, axis=1))

    # Compute explanations
    if exp == 'gradient':
        grads = _get_grads(model)
        np.save(f'{directory}/grads_{mode_connect}{perturb}{i}.npy', grads)
    elif exp == 'smoothgrad':
        sg = _get_sg(model)
        np.save(f'{directory}/sg_{mode_connect}{perturb}{i}.npy', sg)
    elif exp == 'shap':
        shaps = _get_shap(model)
        np.save(f'{directory}/shaps_{mode_connect}{perturb}{i}.npy', shaps)
    else:
        pass  # TODO: implement other explanations
    #print(f"workhorse {i} complete")

def _load_config(config_file, name):
    """Load config from file"""
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config[name]

def _load_model(idx):
    """Load model from globals (model_class, model_args, directory, mode_perturb_args, ts)"""
    # Load perturbed model (implement perturbations for mode connectivity)
    if mode_connect:
        model = curves.CurveNet(*curve_args)
        state_dict = torch.load(f'{directory}/{curve_type}_{idx}.pth')
        model.load_state_dict(state_dict)
        if perturb:
            model = curves.CurveNetPerturb(model, TabularModel,
                                           TabularModelPerturb,
                                           mode_perturb_args, ts=ts)
    else:
        model = model_class(*model_args)
        state_dict = torch.load(f'{directory}/model_{idx}.pth')
        model.load_state_dict(state_dict)
        if perturb:
            model = TabularModelPerturb(model, n_weight_perturbations,
                                        weight_sigmas, weight_layers)  # No FMNIST yet
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model

def _get_logits(model):
    """Get logits from globals (model, model_class, X_test_full, mode_connect, perturb)"""
    if mode_connect:
        if perturb:
            logits = model.compute_logits(X_test_full)
        else:
            logits = model.compute_logits(X_test_full, model_class, ts).mean(axis=0)
    else:
        if perturb:
            logits = model.forward(torch.FloatTensor(X_test_full)).detach().numpy().mean(axis=0)
        else:
            logits = model.forward(torch.FloatTensor(X_test_full)).detach().numpy()
    return logits

def _get_grads(model):
    if mode_connect:
        if perturb:
            grads = model.compute_gradients(X_test)
        else:
            grads = model.compute_gradients(X_test, model_class, ts).mean(axis=0)
    else:
        if perturb:
            grads = model.compute_gradients(X_test, mean=True)
        else:
            grads = model.compute_gradients(X_test, return_numpy=True)
    return grads

def _get_sg(model):
    if mode_connect:
        if perturb:
            sg = model.compute_gradients(noisy_x)
            sg = sg.reshape(n_input_perturbations, n_inputs, n_features).mean(axis=0)
        else:
            sg = model.compute_gradients(noisy_x, model_class, ts).mean(axis=0)
            sg = sg.reshape(n_input_perturbations, n_inputs, n_features).mean(axis=0)
    else:
        if perturb:
            sg = model.compute_gradients(torch.FloatTensor(noisy_x), mean=True)
            sg = sg.reshape(n_input_perturbations, n_inputs, n_features).mean(axis=0)
        else:
            sg = model.compute_gradients(torch.FloatTensor(noisy_x), return_numpy=True)
            sg = sg.reshape(n_input_perturbations, n_inputs, n_features).mean(axis=0)
    return sg

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='heloc', help='dataset name')
    parser.add_argument('--loo', action='store_true', help='leave-one-out as source of randomness')
    parser.add_argument('--preds', action='store_true', help='save predictions')
    parser.add_argument('--logits', action='store_true', help='save logits')
    parser.add_argument('--explanation', type=str, default='', help='save explanations using config file parameters (gradient, smoothgrad, etc.)')
    parser.add_argument('--mode_connect', action='store_true', help='load mode connected models')
    parser.add_argument('--perturb', action='store_true', help='perturb weights and save mean results')
    parser.add_argument('--config', type=str, default='postprocess_config.json', help='config file to load')
    parser.add_argument('--parallel', action='store_true', help='run in parallel')

    # Get config dictionary
    args = vars(parser.parse_args())
    name = args['name']
    config = _load_config(args['config'], name)
    exp = args['explanation']
    if exp != '':
        exp_params = config['explanations'][exp]
        print(bold("Explanation:"), exp)
        print(bold("Explanation parameters:"), exp_params)
    
    # Load dataset
    print(bold("Dataset:"), name)
    trainset, testset = datasets.load_dataset(name)
    X_test_full, y_test_full = testset.data.numpy(), testset.labels.numpy()
    if name in ['default', 'fmnist', 'gmsc', 'adult', 'heloc']:
        X_test, y_test = X_test_full[:1000], y_test_full[:1000]
        if exp != '':
            print("Computing explanation metrics for first 1000 test points")
    else:
        X_test, y_test = X_test_full, y_test_full
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
    elif name in datasets._tabular_datasets:
        model_args = [n_features, datasets.tabular.layers[name]]

    # Perturbation
    perturb = 'perturb_' if args['perturb'] else ''
    if perturb:
        n_weight_perturbations = config['perturb']['n_weight_perturbations']
        weight_sigmas = config['perturb']['weight_sigmas']
        weight_layers = config['perturb']['weight_layers']

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
        if perturb:
            n_pert_mode = config['mode_connect']['n_curve_perturbations']
            sigmas_pert = config['mode_connect']['curve_perturb_sigmas']
            pert_layers = config['mode_connect']['curve_perturb_layers']
            mode_perturb_args = [n_pert_mode, sigmas_pert, pert_layers]

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
        if perturb and not mode_connect:
            print(bold("Perturbing weights"))
            print(bold("Number of weight perturbations:"), n_weight_perturbations)
            print(bold("Weight sigmas:"), weight_sigmas)
            print(bold("Weight layers:"), weight_layers)
        if mode_connect:
            print(bold("Mode connectivity"))
            print(bold("Curve type:"), curve_type)
            print(bold("Number of curve samples:"), n_curve_samples)
            if perturb:
                print(bold("Number of curve perturbations:"), n_pert_mode)
                print(bold("Curve perturb sigmas:"), sigmas_pert)
                print(bold("Curve perturb layers:"), pert_layers)

        # Fixed smoothgrad perturbations
        if exp == 'smoothgrad':
            print(bold("Fixed smoothgrad perturbations"))
            n_input_perturbations = config['explanations']['smoothgrad']['n_input_perturbations']
            sg_sigma = config['explanations']['smoothgrad']['input_sigma']
            print(bold("Number of smoothgrad perturbations:"), n_input_perturbations)
            print(bold("Smoothgrad sigma:"), sg_sigma)
            np.random.seed(0)
            noise = np.random.normal(scale=sg_sigma,
                                     size=(n_input_perturbations,
                                           n_inputs, n_features))
            noisy_x = np.vstack([np.expand_dims(X_test, axis=0)] * n_input_perturbations) + noise
            noisy_x = noisy_x.reshape(-1, n_features)


        # Run in parallel
        if args['parallel']:
            print(bold("Running in parallel"))
            start_time = time.time()
            set_start_method('spawn')
            num_cores = cpu_count()
            print(bold("Number of cores:"), num_cores)
            with tqdm_joblib(tqdm(desc="Computing Statistics", total=config['n'])) as progress_bar:
                Parallel(n_jobs=num_cores)(delayed(_get_model_stats)(i) for i in range(config['n']))
            print(bold(f"Total time: {time.time() - start_time} seconds"))
        
        else:
            # Run sequentially
            print(bold("Running sequentially"))
            start_time = time.time()
            for i in tqdm(range(config['n']), desc="Computing Statistics"):
                _get_model_stats(i)
            print(bold(f"Total time: {time.time() - start_time} seconds"))