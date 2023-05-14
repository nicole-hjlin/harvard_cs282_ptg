import datasets.fmnist as fmnist
import datasets.tabular as tabular

_tabular_datasets = ['german', 'adult', 'heloc', 'default', 'gmsc']

def load_dataset(name):
    if name == 'fmnist':
        return fmnist.load_fmnist_dataset()
    elif name in _tabular_datasets:
        return tabular.load_tabular_dataset(name)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
def get_model_class(name: str):
    """Returns a model class by dataset name"""
    if name == 'fmnist':
        model_class = fmnist.LeNet5
    elif name in _tabular_datasets:
        model_class = tabular.TabularModel
    else:
        raise ValueError(f'Unknown dataset name: {name}')
    return model_class

def get_curve_class(name: str):
    """Returns a curve model class by dataset name"""
    if name == 'fmnist':
        curve_class = fmnist.LeNet5Curve
    elif name in _tabular_datasets:
        curve_class = tabular.TabularModelCurve
    else:
        raise ValueError(f'Unknown dataset name: {name}')
    return curve_class

def get_learning_pipeline(name: str):
    """Returns a learning pipeline by dataset name"""
    if name == 'fmnist':
        learning_pipeline = fmnist.learning_pipeline
    elif name in _tabular_datasets:
        learning_pipeline = tabular.learning_pipeline
    else:
        raise ValueError(f'Unknown dataset name: {name}')
    return learning_pipeline