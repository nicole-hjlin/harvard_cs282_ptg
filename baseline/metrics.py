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
    # May run into memory issues in future
    preds = torch.zeros((len(g), x.shape[0]), dtype=torch.int)
    for i, model in enumerate(tqdm(g)):
        # predict function is flexible to type(x)
        preds[i] = model.predict(x, return_numpy=False)

    # return shape is (no. models, no.inputs)
    # may want to flatten in case of single input
    if return_numpy:
        return preds.numpy()
    return preds

def compute_accuracies(
    preds: torch.Tensor | np.ndarray,
    y: torch.Tensor | np.ndarray,
):
    # Convert to numpy if a tensor
    preds = convert_to_numpy(preds)
    y = convert_to_numpy(y)
        
    return (preds==y).mean(axis=1)*100

def compute_abstention_rate(
    preds: torch.Tensor | np.ndarray,
):
    return np.mean(preds==np.inf, axis=1)*100

def compute_disagreement(
    preds: torch.Tensor | np.ndarray,
):
    """Proportion of inputs with disagreement between at least one pair of models
    For each input in x_test, if ALL predictions ARE NOT the same, assign 1 to that input
    Return "mean of portion of test data with p_flip > 0"
    Note that we're not computing the probability of flip between pairwise models
    For now, we're just computing if the probability is greater than 0
    i.e. at least one model has a different prediction to all the others"""
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
    ensembles: list[list[nn.Module]],  # list of g, where g is list of singletons OR list of ensemble class
    X_test: torch.Tensor,  # multiple inputs
    a: float,
):
    # Ideally we just have one function that takes in a list of models
    # So we might need a model class for the selective ensemble
    # We also need a function for the standard ensemble

    # Alternatively, modify util/select_preds() to process a whole array of inputs instead of just 1
    # As the inputs above suggest
    # select_preds should return array of size no. inputs, with nan value for abstention
    # e.g. size of preds is 10 x 10000 for FMNIST (10 ensembles, 10000 test points)
    preds = torch.stack([select_preds(g, a, X_test) for g in ensembles])
    p_flip_positive = torch.any(preds!=preds[0], dim=1).to(torch.float)

    # We'll assume that if all ensembles abstain for one particular input, all predictions are the same for that input

    return p_flip_positive.mean()

def SSIM(
    grads1: torch.Tensor,
    grads2: torch.Tensor,
):
    # Structural Similarity (SSIM), varies from -1 to 1
    # For FMNIST, this replaces Spearman's rank
    # They use: https://scikit-image.org/docs/stable/auto_examples/transform/plot_ssim.html

    # For now, let's implement it for 2 gradient vectors that are the "feature attributions" for a single input

    return -1


def pearsonr(
    grads1: torch.Tensor,
    grads2: torch.Tensor,
):
    # I think we'll literally just use:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
    return -1