"""Metrics for evaluating the performance of a model or ensemble of models"""

import torch
from torch import nn
from util import select_preds, convert_to_numpy
import numpy as np
from tqdm import tqdm

def compute_preds(
    g: list[nn.Module],
    x: torch.Tensor | np.ndarray,
    return_numpy: bool = False,
):
    """Compute predictions for each model in g for each input in x"""

    # May run into memory issues for larger datasets
    preds = torch.zeros((len(g), x.shape[0]), dtype=torch.int)
    for i, model in enumerate(tqdm(g)):
        preds[i] = model.predict(x, return_numpy=False) # flexible to type(x)

    # return shape is (no. models, no.inputs)
    # may want to flatten in case of single input
    if return_numpy:
        return preds.numpy()
    return preds

def compute_accuracies(
    preds: torch.Tensor | np.ndarray,
    y: torch.Tensor | np.ndarray,
):
    """Compute accuracy for each model for each prediction"""

    # Convert to numpy if tensors
    preds = convert_to_numpy(preds)
    y = convert_to_numpy(y)

    return (preds==y).mean(axis=1)*100


def compute_abstention_rate(
    preds: torch.Tensor | np.ndarray,
):
    """Compute the percentage of inputs for which an ensemble abstains"""

    return np.mean(preds==np.inf, axis=1)*100


def compute_disagreement(
    preds: torch.Tensor | np.ndarray,
):
    """Proportion of inputs with disagreement between at least one pair of models
    For each input, if all predictions are not the same between models,
    assign 1 to that input, else assign 0 since all models agree
    preds is a 2D array of size (no. models, no. inputs)
    or a 1D array, in which case 0.0 is returned"""

    # Special case for predictions from a single model
    if len(preds.shape)==1:
        return 0.0  # no disagreement

    # Convert to numpy if a tensor
    preds = convert_to_numpy(preds)

    # For each input, check if any models predict different to the first model
    p_flip_positive = np.any(preds!=preds[0], axis=0)  # size is no. inputs

    return p_flip_positive.mean()


def compute_ensemble_predictions(
    ensemble_size: int,
    n_ensembles: int,
    preds: torch.Tensor | np.ndarray,
    ensemble_method: str = 'selective',
):
    """Compute predictions for each selective ensemble
    preds is a 2D array of size (no. models, no. inputs)"""

    # Convert to numpy if a tensor
    preds = convert_to_numpy(preds)

    # Compute no. inputs
    n_inputs = preds.shape[1]

    # Initialize predictions for each selective ensemble
    ensemble_preds = np.zeros((n_ensembles, n_inputs))

    # Compute predictions for each selective ensemble
    for i in range(n_ensembles):
        # Aggregate predictions for one ensemble
        ensemble_model_preds = preds[ensemble_size*i:ensemble_size*(i+1)]

        # Combine predictions
        ensemble_preds[i] = select_preds(ensemble_model_preds, ensemble_method=ensemble_method)

    # Return predictions for each selective ensemble
    return ensemble_preds


def p_flip_ensemble(
    ensembles: list[list[nn.Module]],
    X_test: torch.Tensor,
    a: float,
):
    """Compute the probability of flip for each ensemble in set of ensembles
    ensembles is a list of g, where g is a list of singletons OR list of ensemble class"""

    # Compute predictions for each ensemble
    preds = torch.stack([select_preds(g, a, X_test) for g in ensembles])
    p_flip_positive = torch.any(preds!=preds[0], dim=1).to(torch.float)

    # We'll assume that if all ensembles abstain for one particular input,
    # all predictions are the same for that input
    return p_flip_positive.mean()
