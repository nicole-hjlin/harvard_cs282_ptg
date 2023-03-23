"""Dataset for tabular data (inherits from torch.utils.data.Dataset)"""
import requests
from os import makedirs, path
import io
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from .german import preprocess_german

download_urls = {'german': 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'}
preprocess_funcs = {'german': preprocess_german}

class TabularDataset(Dataset):
    """
    A dataset for tabular data, instantiated for a specific dataset as a train or test set.
    """
    def __init__(self, name, download_url, download=True, train=True, preprocess_func=None,
                 sep=" +", transform=ToTensor()):
        self.name = name
        self.download_url = download_url
        self.train = train
        self.preprocess_func = preprocess_func
        self.sep = sep
        self.transform = transform
        self.data_dir = f'../data/{name}'
        self.train_file = f'{self.data_dir}/{name}_train.csv'
        self.test_file = f'{self.data_dir}/{name}_test.csv'
        
        if download:
            self.download_and_process()
        
        self.data = self.load_data()

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
            response = requests.get(self.download_url)
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
        """Load the train or test set into a pandas DataFrame."""
        data_file = self.train_file if self.train else self.test_file
        data = pd.read_csv(data_file)
        return data

    def __len__(self):
        """Return the length of the dataset. Necessary for PyTorch's DataLoader."""
        return len(self.data)

    def __getitem__(self, idx):
        """Return the sample at index idx. Necessary for PyTorch's DataLoader."""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data.iloc[idx]
        return sample


def load_tabular_dataset(name: str) -> tuple[TabularDataset, TabularDataset]:
    """Load tabular dataset, return train and test sets
    If neither train nor test file exists,
    download the dataset, preprocess it,
    and split it into train and test sets,
    overwriting any existing files
    If both train and test files exist,
    load them from the '../data/name/' directory
    """

    # Load trainset
    trainset = TabularDataset(
        name=name,
        download_url=download_urls[name],
        download=True,
        train=True,
        preprocess_func=preprocess_funcs[name],
    )

    # Load testset
    testset = TabularDataset(
        name='german',
        download_url=download_urls[name],
        download=True,
        train=False,
        preprocess_func=preprocess_funcs[name],
    )

    # Return train/test sets
    return trainset, testset