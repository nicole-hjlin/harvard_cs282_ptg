"""Dataset for tabular data (inherits from torch.utils.data.Dataset)"""
import requests
from os import makedirs, path
import io
import wandb
from torch.utils.data import DataLoader
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
from torchvision.transforms import ToTensor
from util import State, get_optimizer
import curves
from .german import preprocess_german
from typing import Tuple


# Would be cleaner if these were included in their respective files actually
download_urls = {'german': 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data',
                 'heloc': None,
                 'default': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls',
                 'gmsc': None,
                 'adult': None}
preprocess_funcs = {'german': preprocess_german,
                    'heloc': None,
                    'default': None,
                    'gmsc': None,
                    'adult': None}
layers = {'german': [128, 64, 16],
          'heloc': [128, 64, 16],
          'moons': [128, 64, 16],
          'default': [128, 64, 16],
          'gmsc': [128, 64, 16],
          'adult': [128, 64, 16]}

def init_curve(S: State):
    """
    Returns a curve object for the learning pipeline
    Not yet implemented for FMNIST
    """
    if S.config['mode_connect'] == 'bezier':
        curve_type = curves.Bezier
    elif S.config['mode_connect'] == 'polychain':
        curve_type = curves.PolyChain
    else:
        raise ValueError(f'Unknown curve type {curve_type}')

    # Create model (with initial parameters)
    input_size, hidden_layers = S.trainset.data.shape[1], layers[S.trainset.name]
    model = curves.CurveNet(
        curve=curve_type,
        architecture=TabularModelCurve,  # only one TabularModel/TabularModelCurve implemented
        num_bends=2,
        input_size=input_size,
        hidden_layers=hidden_layers,
        fix_start=False,
        fix_end=False,
    )

    # Load initial parameters
    for i, seed in enumerate([S.seed, S.seed+S.config['n']]):  # use convention of seed, n+seed
        torch.manual_seed(seed)
        init_model = S.net(input_size, hidden_layers)
        model.import_base_parameters(init_model, i)
    
    # Reset seed appropriately
    torch.manual_seed(S.seed)

    # Return curve
    return model


def learning_pipeline(S) -> nn.Module:
    """Learning pipeline for tabular datasets
    Currently assumes binary classification
    """

    # Input size, layers
    input_size = S.trainset.data.shape[1]
    layer = layers[S.trainset.name]

    # Ensure mode_connect is not used with loo
    if S.config['mode_connect'] != '' and S.config['loo']:
        raise ValueError("mode_connect not supported with loo")

    # Random seed (controls initialization and optimizer stochasticity)
    if S.config['loo']:
        torch.manual_seed(0)
        print("Training with seed 0 (loo)")
        print("Subset length:", len(S.trainset))
        print("Subset labels mean:", S.trainset.labels.float().mean().item())
        print("Run", S.seed)
    else:
        torch.manual_seed(S.seed)
        print("Training with seed", S.seed)
        print("Run", S.seed)
    
    # Initialize model (binary classification) and set to train mode
    if S.config['mode_connect'] == '':
        S.net = S.net(input_size, layer)
    else:
        S.net = init_curve(S)
    S.net.train()

    # Set up loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = get_optimizer(S)

    # Set up dataloader (NB shuffle=False by default, we ignore effects of shuffling)
    loader = DataLoader(
        S.trainset,
        batch_size=S.config['batch_size'],
    )

    use_wandb = S.config['wandb']
    if use_wandb:
        wandb.watch(S.net, loss_fn, log='all', log_freq=1)
    epochs = S.config['epochs']

    with tqdm(total=epochs*len(loader)) as pbar:
        for _ in range(epochs):
            for i, (x, y) in enumerate(loader):
                optimizer.zero_grad()
                y_pred = S.net(x)
                loss = loss_fn(y_pred, y)
                loss.backward()
                optimizer.step()
                if use_wandb:
                    if i % 100 == 0:
                        acc = (y_pred.argmax(-1) == y).float().mean()
                        # Log metrics to wandb
                        wandb.log({
                            'loss': loss,
                            'acc': acc,
                        })
                pbar.update(1)

    # Quick train/test evaluation
    if not use_wandb:
        S.net.eval()
        y_pred, y_pred_te = S.net(S.trainset.data), S.net(S.testset.data)
        y, y_te = S.trainset.labels, S.testset.labels
        loss, loss_te = loss_fn(y_pred, y), loss_fn(y_pred_te, y_te)
        acc = (y_pred.argmax(-1) == y).float().mean()
        acc_te = (y_pred_te.argmax(-1) == y_te).float().mean()
        acc, acc_te = round(acc.item(), 3), round(acc_te.item(), 3)
        loss, loss_te = round(loss.item(), 3), round(loss_te.item(), 3)
        print('Train/Test Accuracy:', acc, acc_te)
        print('Train/Test Loss:', loss, loss_te)

    # Return the trained model
    return S.net


class TabularSubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        
        # Assign the attributes from the original dataset to the new subset
        self.name = dataset.name
        self.data = dataset.data[indices]
        self.labels = dataset.labels[indices]


class TabularDataset(Dataset):
    """
    A dataset for tabular data, instantiated for a specific dataset as a train or test set.
    """
    def __init__(self, name, download_url=None, train=True, preprocess_func=None,
                 sep=" +", transform=ToTensor()):
        self.name = name
        self.download_url = download_url
        self.train = train
        self.preprocess_func = preprocess_func
        self.sep = sep
        self.transform = transform
        self.data_dir = f'datasets/data/{name}'
        self.train_file = f'{self.data_dir}/{name}_train.csv'
        self.test_file = f'{self.data_dir}/{name}_test.csv'

        if download_url is not None:
            self.download_and_process()

        self.data, self.labels = self.load_data()

    def download_and_process(self):
        """
        If the directory doesn't exist, make it.
        If the train or test file doesn't exist,
        download the dataset, preprocess it,
        and split it into train and test sets,
        Save them as CSV files.
        """
        makedirs(self.data_dir, exist_ok=True)

        if not path.isfile(self.train_file) or not path.isfile(self.test_file):
            # Download the dataset
            response = requests.get(self.download_url, timeout=60)
            response.raise_for_status()
            raw_data = pd.read_csv(io.StringIO(response.text),
                                   sep=self.sep,
                                   engine="python",
            )

            # Preprocess the dataset if a preprocessing function is provided
            raw_data = self.preprocess_func(raw_data)

            # Split the dataset into train and test sets
            train_data, test_data = train_test_split(raw_data, test_size=0.2, random_state=0)

            # Save the train and test sets as CSV files
            train_data.to_csv(self.train_file, index=False)
            test_data.to_csv(self.test_file, index=False)

    def load_data(self):
        """Load the train or test set into a pandas DataFrame and convert to tensors."""
        data_file = self.train_file if self.train else self.test_file
        data = pd.read_csv(data_file)

        # Assuming the labels column is called 'label'
        labels = data['label']
        data = data.drop(columns=['label'])
        self.feature_names = data.columns

        # Retrieve values as numpy arrays
        data = data.values
        labels = labels.values

        # Convert the remaining data to tensors
        data = torch.tensor(data, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long)

        return data, labels

    def __len__(self):
        """Return the length of the dataset. Necessary for PyTorch's DataLoader."""
        return len(self.data)

    def __getitem__(self, idx):
        """Return the sample at index idx. Necessary for PyTorch's DataLoader."""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label


def load_tabular_dataset(name: str) -> Tuple[TabularDataset, TabularDataset]:
    """Load tabular dataset, return train and test sets
    If neither train nor test file exists,
    download the dataset, preprocess it,
    and split it into train and test sets,
    overwriting any existing files
    If both train and test files exist,
    load them from the '../data/name/' directory

    Called once in main.py
    """

    # Load trainset
    trainset = TabularDataset(
        name=name,
        download_url=download_urls[name],
        train=True,
        preprocess_func=preprocess_funcs[name],
    )

    # Load testset
    testset = TabularDataset(
        name=name,
        download_url=download_urls[name],
        train=False,
        preprocess_func=preprocess_funcs[name],
    )

    # Return train/test sets
    return trainset, testset


class TabularModel(nn.Module):
    """Tabular model for binary classification"""
    def __init__(self, input_size, hidden_layers):
        super(TabularModel, self).__init__()
        self.input_size = input_size
        self.hidden_layers = hidden_layers

        model_layers = []
        previous_layer_size = input_size
        for layer_size in hidden_layers:
            model_layers.append(nn.Linear(previous_layer_size, layer_size))
            model_layers.append(nn.ReLU())
            previous_layer_size = layer_size
        model_layers.append(nn.Linear(previous_layer_size, 2))
        model_layers.append(nn.Softmax(dim=-1))
        self.network = nn.Sequential(*model_layers)

    def forward(self, x):
        """Forward pass of the model (softmax output)"""
        return self.network(x)

    def predict(self, x, return_numpy=False):
        """Predict method, returns hard predictions
        Flexible to take in tensor or numpy array
        Shape of x is (no. inputs, no. features)
        Returns numpy array if return_numpy is True"""

        # Convert input to tensor if it's a numpy array
        x = convert_to_tensor(x)

        # Forward pass and argmax for hard prediction
        self.eval()
        with torch.no_grad():  # save memory (inference only)
            preds = self(x.float()).argmax(dim=1)  # softmax output

        # Return hard predictions
        if return_numpy:
            return preds.detach().numpy()
        return preds

    def compute_gradients(self, x, softmax=False, label=1,
                          return_numpy=False):
        """Compute gradients of the model with respect to the input
        Flexible to take in tensor or numpy array
        Shape of x is (no. inputs, no. features) or (no. features)
        Returns tensor or numpy array depending on return_numpy"""

        # Convert input to tensor if it's a numpy array
        x = convert_to_tensor(x)

        # If single input, add batch dimension
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        # Compute gradients
        self.eval()
        x = x.float()
        x.requires_grad = True
        if softmax:
            out = self(x)[:, label]
        else:
            out = self.network[:-1](x)[:, label]

        grads = torch.autograd.grad(outputs=out, inputs=x,
                                    grad_outputs=torch.ones_like(out))[0]


        # Convert to numpy array if return_numpy is True
        if return_numpy:
            grads = grads.detach().numpy()

        return grads
    
    def compute_smoothgrad(self, x):
        """Compute SmoothGrad of the model with respect to the input
        Flexible to take in tensor or numpy array
        Shape of x is (no. inputs, no. features) or (no. features)
        Returns tensor or numpy array depending on return_numpy"""

        # Convert input to tensor if it's a numpy array
        x = convert_to_tensor(x)

        # If single input, add batch dimension
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        # Compute SmoothGrad
        self.eval()
        x = x.float()
        x.requires_grad = True
        out = self(x)[:, 1]
        grads = torch.autograd.grad(outputs=out, inputs=x,
                                    grad_outputs=torch.ones_like(out))[0]
        grads = grads.detach().numpy()
        return grads
    
    def compute_perturbed_stats(self, x, x_full=None, n=100, sigmas=[0.5],
                                perturb_layers=['network.0.weight']):
        if x_full is None:
            x_full = x
        perturbed_model = TabularModelPerturb(self, n, sigmas=sigmas,
                                              perturb_layers=perturb_layers)
        grads = perturbed_model.compute_gradients(x)
        preds = perturbed_model.predict(x_full)
        return grads, preds


class TabularModelPerturb(nn.Module):
    def __init__(self, base_model, num_perturbations, sigmas=[0.1],
                 perturb_layers='all', train=None):
        super().__init__()
        self.base_model = base_model
        self.num_perturbations = num_perturbations
        if perturb_layers == 'all':
            self.perturb_layers = list(base_model.state_dict().keys())
        else:
            self.perturb_layers = perturb_layers
        self.sigmas = sigmas if isinstance(sigmas, list) else [sigmas] * len(self.perturb_layers)

        self.models = nn.ModuleList()
        self.scalars = self.compute_scalars(train)
        for i in range(num_perturbations):
            model = TabularModel(base_model.input_size, base_model.hidden_layers)
            model.load_state_dict(base_model.state_dict())
            with torch.no_grad():
                for j, layer_name in enumerate(self.perturb_layers):
                    layer_weights = model.state_dict()[layer_name]
                    #torch.manual_seed(i)  # reproducibility
                    noise = torch.randn_like(layer_weights)
                    noise *= self.scalars[j] * self.sigmas[j]
                    layer_weights.add_(noise)
            self.models.append(model)

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return torch.stack(outputs, dim=0).mean(axis=0)
    
    def compute_scalars(self, train):
        if train is None:
            scalars = [1] * len(self.perturb_layers)
            # for layer_name in self.perturb_layers:
            #     layer_weights = self.base_model.state_dict()[layer_name]
            #     d = layer_weights.flatten().shape[0]
            #     scalars.append(1 / np.sqrt(d))
            return scalars
        X_train, y_train = train
        X_train.requires_grad = True
        self.base_model.eval()
        self.base_model.zero_grad()
        y_pred = self.base_model(X_train)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(y_pred, y_train)
        loss.backward()

        # Compute grad_squared for each layer
        grad_vec = []
        for name, param in self.base_model.named_parameters():
            if name in self.perturb_layers:
                grad_vec.append(1/((param.grad**2)+1e-16))

        # This whole process is to avoid overflow (might be cleaner to flatten)
        norm = sum([(grad_vec[i]**2).sum() for i in range(len(grad_vec))]).sqrt()
        grad_vec = [grad_vec[i] / norm for i in range(len(grad_vec))]

        # # Take reciprocal of scalars
        # scalars = [1/g for g in grad_vec]
        # # Replace inf with max value
        # for i in range(len(scalars)):
        #     scalars[i][torch.isinf(scalars[i])] = scalars[i][~torch.isinf(scalars[i])].max()
        # # Normalize scalars
        # squares_inv = sum([(scalars[i]**2).sum() for i in range(len(scalars))])
        # scalars = [scalars[i] / squares_inv.sqrt() for i in range(len(scalars))]

        return [g**0.5 for g in grad_vec]

    def predict(self, x, mean=True):
        logits = self(torch.FloatTensor(x))
        if mean:
            preds = logits.mean(axis=0).argmax(-1)
        else:
            preds = logits.argmax(-1)
        return preds.detach().numpy()
    
    def compute_gradients(self, x, mean=True):
        grads = [model.compute_gradients(x, return_numpy=True) for model in self.models]
        if mean:
            return np.array(grads).mean(axis=0)
        return np.array(grads)
    
    def compute_logits(self, x, mean=True):
        logits = [model(x).detach().numpy() for model in self.models]
        if mean:
            return np.array(logits).mean(axis=0)
        return np.array(logits)


class TabularModelCurve(nn.Module):
    """Tabular curve model for binary classification"""
    def __init__(self, input_size, hidden_layers, fix_points):
        super(TabularModelCurve, self).__init__()
        self.input_size = input_size
        self.hidden_layers = hidden_layers

        model_layers = []
        previous_layer_size = input_size
        for layer_size in hidden_layers:
            model_layers.append(curves.Linear(previous_layer_size, layer_size, fix_points=fix_points))
            model_layers.append(nn.ReLU())
            previous_layer_size = layer_size
        model_layers.append(curves.Linear(previous_layer_size, 2, fix_points=fix_points))
        model_layers.append(nn.Softmax(dim=-1))
        self.network = nn.Sequential(*model_layers)

    def forward(self, x, coeffs_t):
        for layer in self.network:
            if isinstance(layer, curves.Linear):
                x = layer(x, coeffs_t)
            else:
                x = layer(x)
        return x


def convert_to_tensor(arr):
    """Conditional conversion to a torch tensor"""

    # Return torch tensor
    return torch.FloatTensor(arr) if isinstance(arr, np.ndarray) else arr


def convert_to_numpy(arr):
    """Conditional conversion to a numpy array"""

    # Return numpy array
    return arr.numpy() if isinstance(arr, torch.Tensor) else arr
