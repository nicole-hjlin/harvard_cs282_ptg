"""Dataset module to load (and download) datasets."""
from .fmnist import load_fmnist_dataset
from .tabular import load_tabular_dataset

def load_dataset(name):
    if name == 'fmnist':
        return load_fmnist_dataset()
    elif name in ['german', 'adult', 'heloc']:
        return load_tabular_dataset(name)
    else:
        raise ValueError(f"Unknown dataset: {name}")