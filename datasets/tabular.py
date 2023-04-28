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
from util import State, convert_to_tensor, get_optimizer
from modconn import curves
from .german import preprocess_german
from typing import Tuple


# Would be cleaner if these were included in their respective files actually
download_urls = {'german': 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data',
                 'heloc': None}
preprocess_funcs = {'german': preprocess_german,
                    'heloc': None}
layers = {'german': [128, 64, 16],
          'heloc': [128, 64, 16]}

def learning_pipeline(S: State) -> nn.Module:
    """Learning pipeline for tabular datasets
    Currently assumes binary classification
    """

    # Input size, layers
    layer = layers[S.trainset.name]
    input_size = S.trainset.data.shape[1]

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
    
    # Initialize model (binary classificaiton) and set to train mode
    S.net = S.net(input_size, layer)
    S.net.train()

    # Set up loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = get_optimizer(S)

    # Set up dataloader
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
                if i % 100 == 0:
                    acc = (y_pred.argmax(-1) == y).float().mean()
                    if use_wandb:
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
        import os, sys
        print(os.getcwd())
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
    
    def compute_perturbed_gradients(self, x, sigma=0.5, n=100):
        perturbed_model = TabularModelPerturb(self, n, sigma)
        return perturbed_model.compute_gradients(x)

class TabularModelPerturb(nn.Module):
    def __init__(self, base_model, num_perturbations, sigma):
        super().__init__()
        self.num_perturbations = num_perturbations

        self.models = nn.ModuleList()
        for i in range(num_perturbations):
            model = TabularModel(base_model.input_size, base_model.hidden_layers)
            model.load_state_dict(base_model.state_dict())
            with torch.no_grad():
                layer_weights = model.state_dict()['network.0.weight']
                #torch.manual_seed(i)  # reproducibility
                noise = torch.randn_like(layer_weights) * sigma
                layer_weights.add_(noise)
            self.models.append(model)

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return torch.stack(outputs, dim=0)
    
    def predict(self, x, mean=True):
        logits = self(torch.FloatTensor(x))
        if mean:
            preds = logits.mean(axis=0).argmax(1)
        else:
            preds = logits.argmax(2)
        return preds.detach().numpy()
    
    def compute_gradients(self, x, mean=True):
        grads = [model.compute_gradients(x, return_numpy=True) for model in self.models]
        if mean:
            return np.array(grads).mean(axis=0)
        return np.array(grads)

class TabularModelCurve(nn.Module):
    """Tabular curve model for binary classification"""
    def __init__(self, input_size, hidden_layers, fix_points: list[bool]):
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
