import torch
from torch import nn
from util import predict_ensemble

def accuracy(
    g: list[nn.Module],
    X_test: torch.Tensor,  # multiple inputs
    y_test: torch.Tensor,
):
    # Mean accuracy over n models
    # Ambiguous in paper- I would evaluate test accuracy?

    # Please someone check/improve this code
    # It's psuedo-code right now
    preds = [model(X_test).argmax(1) for model in g]
    accs = [(pred==y_test).mean()*100 for pred in preds]
    return accs.mean(), accs.std()

def p_flip_singleton(
    g: list[nn.module],  # 200 models for FMNIST
    X_test: torch.Tensor,  # multiple inputs
):
    # Percentage of inputs with disagreement between at least one pair of models
    # For each input in x_test, if ALL predictions ARE NOT the same, assign 1 to that input
    # Return "mean of portion of test data with p_flip > 0"

    # Note that we're not computing the probability of flip between pairwise models
    # For now, we're just computing if the probability is greater than 0
    # i.e. at least one model has a different prediction to all the others
    preds = torch.stack([model(X_test).argmax(1) for model in g])

    # e.g. size of preds is 200 x 10000
    # for each input, check if any models predict different to the first model
    p_flip_positive = torch.any(preds!=preds[0], dim=1).to(torch.float)

    # Please someone check/improve this code
    # It seems to work if size of preds is no.models x no.inputs
    # Maybe the preds line fails on certain types
    # For FMNIST we need to do x_test.unsqueeze(1) I think
    
    return p_flip_positive.mean()

def p_flip_ensemble(
    ensembles: list[list[nn.module]],  # list of g, where g is list of singletons OR list of ensemble class
    X_test: torch.Tensor,  # multiple inputs
    a: float,
):
    # Ideally we just have one function that takes in a list of models
    # So we might need a model class for the selective ensemble
    # We also need a function for the standard ensemble

    # Alternatively, modify util/predict_ensemble() to process a whole array of inputs instead of just 1
    # As the inputs above suggest
    # predict_ensemble should return array of size no. inputs, with nan value for abstention
    # e.g. size of preds is 10 x 10000 for FMNIST (10 ensembles, 10000 test points)
    preds = torch.stack([predict_ensemble(g, a, X_test) for g in ensembles])
    p_flip_positive = torch.any(preds!=preds[0], dim=1).to(torch.float)

    # We'll assume that if all ensembles abstain for one particular input, all predictions are the same for that input

    return p_flip_positive.mean()

def accuracy_ensemble():
    # probably easier to just make an ensemble class so it can be passed as a "single model" to accuracy()
    # restructure as you wish, we just need to evaluate selective ensembles and standard ensembles (average?)
    # take in an ensemble and return the mean accuracy, std. dev, and abstention rate
    return -1

def SSIM(
    grads1: torch.Tensor,
    grads2: torch.Tensor,
):
    # Structural Similarity (SSIM), varies from -1 to 1
    # For FMNIST, this replaces Spearman's rank
    # They use: https://scikit-image.org/docs/stable/auto_examples/transform/plot_ssim.html

    # For now, let's implement it for 2 gradient vectors that are the "feature attributions" for a single input

    return -1