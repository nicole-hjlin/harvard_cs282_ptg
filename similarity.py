import numpy as np
from tqdm import tqdm
from util import get_statistics
from style import bold
import datasets
import argparse
from postprocess import tqdm_joblib, _load_config
from multiprocessing import set_start_method, cpu_count
from joblib import Parallel, delayed
from tqdm import tqdm

def get_top_k(k, X, return_sign=False):
    top_k = np.argsort(np.abs(X), axis=-1)[..., -k:][..., ::-1]

    if return_sign:
        # Get the sign of each feature in the top_k
        idx = np.indices(top_k.shape)
        indexing_tuple = tuple(idx[:-1]) + (top_k,)
        sign_vector = np.sign(X[indexing_tuple])
        return top_k, sign_vector.astype(int)
    else:
        return top_k
    
def average_ground_truth_score(X, signs_X, gt, signs_gt, metric_func):
    """
    Compute the average similarity score between the ground truth
    and the ensemble, across all trials (each trial is one ensemble size).
    Input size is (n_trials, n_inputs, k)
    Ground truth, gt, is size (n_inputs, k)
    Returns a vector of size (n_inputs,)
    where each entry is the average similarity score
    """
    n_trials, n_inputs, k = X.shape
    scores = np.zeros(n_inputs)

    for i in range(n_trials):
        scores += metric_func(X[i], gt, signs_X[i], signs_gt)

    scores /= n_trials

    return scores

def ground_truth_score(X, signs_x, gt, signs_gt, metric_func):
    """
    As in average_ground_truth_score
    but returns a vector of size (n_trials, n_inputs)
    """
    n_trials, n_inputs, k = X.shape
    scores = np.zeros((n_trials, n_inputs))

    for i in range(n_trials):
        scores[i] = metric_func(X[i], gt, signs_x[i], signs_gt)

    return scores

def average_pairwise_score(X, signs_X, metric_func):
    """Compute the average pairwise score across trials.
    Input size is (n_trials, n_inputs, k)
    Returns a vector of size (n_inputs,)
    where each entry is the average pairwise score of the
    top-k features across trials for that input.
    Score is computed using metric_func e.g. top_k_cdc"""
    n_models, n_inputs, k = X.shape
    pairwise_scores = np.zeros(n_inputs)

    for i in range(n_models):
        for j in range(i+1, n_models):
            scores = metric_func(X[i], X[j], signs_X[i], signs_X[j])
            pairwise_scores += scores

    total_pairs = n_models * (n_models - 1) / 2
    pairwise_scores /= total_pairs

    return pairwise_scores

def top_k_sa(x, y, signs_x, signs_y):
    x_signed = x * signs_x
    y_signed = y * signs_y
    n_inputs, k = x.shape
    sa_scores = np.zeros(n_inputs)

    for i in range(n_inputs):
        shared_features = np.intersect1d(x_signed[i], y_signed[i])
        sa_scores[i] = len(shared_features) / k

    return sa_scores

def top_k_sa_WIP(x, y, signs_x, signs_y):
    # Work in progress faster version of top_k_sa
    x_signed = x * signs_x
    y_signed = y * signs_y

    # Expand dimensions for broadcasting
    x_signed_expanded = x_signed[:, :, np.newaxis]
    y_signed_expanded = y_signed[:, np.newaxis, :]

    # Compute common signed features for each input
    common_signed_features = np.sum(x_signed_expanded == y_signed_expanded, axis=-1)

    # Calculate the Sign Agreement scores
    sa_scores = common_signed_features.max(axis=-1) / x.shape[1]

    return sa_scores

def top_k_cdc_readable(x, y, signs_x, signs_y):
    # binary metric: 0 if any
    # x and y are n_inputs x k arrays (top-k per input)
    # signs_x and signs_y are n_inputs x k arrays (signs of top-k per input)
    # we want to return a n-dimensional array where
    # each entry is the consistency between x and y
    n_inputs, k = x.shape
    cdc_scores = np.ones(n_inputs)

    for i in range(n_inputs):
        for j in range(k):
            feature_x = x[i, j]
            if feature_x in y[i]:
                index_y = np.where(y[i] == feature_x)[0][0]

                if signs_x[i, j] != signs_y[i, index_y]:
                    cdc_scores[i] = 0
                    break

    return cdc_scores

def top_k_cdc(x, y, signs_x, signs_y):
    # Expand dimensions for broadcasting
    x_expanded = x[:, :, np.newaxis]
    y_expanded = y[:, np.newaxis, :]
    signs_x_expanded = signs_x[:, :, np.newaxis]
    signs_y_expanded = signs_y[:, np.newaxis, :]

    # Find common features with different signs
    common_features_diff_signs = np.logical_and(x_expanded == y_expanded, signs_x_expanded != signs_y_expanded)

    # Check if any common feature with different sign exists for each input
    cdc_scores = np.logical_not(np.any(common_features_diff_signs, axis=(1, 2)))

    return cdc_scores.astype(float)

def top_k_ssa(x, y, signs_x, signs_y):
    """
    Returns Signed Set *Agreement* (i.e., 1 means perfect agreement)
    x and y are n_inputs x k arrays (top-k per input)
    signs_x and signs_y are n_inputs x k arrays (signs of top-k per input)
    cdc has size n_inputs
    scores has size n_inputs
    """ 
    # First, we need to satisfy CDC, so check that first:
    # limited_sx = signs_x[np.arange(x.shape[0])[:,None], x]
    # limited_sy = signs_y[np.arange(x.shape[0])[:,None], x]
    # xeq = (limited_sx == limited_sy)#  + (-1) * (limited_sx != limited_sy)
    # limited_sx = signs_x[np.arange(y.shape[0])[:,None], y]
    # limited_sy = signs_y[np.arange(y.shape[0])[:,None], y]
    # yeq = (limited_sx == limited_sy)
    # cdc = np.logical_and(np.all(xeq == 1, axis=1), np.all(yeq == 1, axis=1))
    cdc = top_k_cdc(x, y, signs_x, signs_y)

    # Next, we need to know whether X and Y have the same top-k features
    k = x.shape[1]
    res = np.zeros([x.shape[0],k])
    for i in range(x.shape[0]):
        for j in range(k):
            if x[i,j] in y[i]:
                res[i,j] = 1
            else:
                res[i,j] = 0
    frac_right = np.sum(res, axis=1)/k
    frac_right = np.where(frac_right == 1, 1, 0)

    # now, frac_right is 1 if x and y have the same top-K features, and cdc is 1 if all of X's top-K features have the same sign in Y
    # so, we need to return 1 if both are true and 0 otherwise
    scores = np.logical_and(frac_right, cdc).astype(float)
    return scores

# Replace this to be similar to top_k_sa, leave pairwise comparisons to average_pairwise_score

def top_k_consistency(top_k):
    """Compute the consistency (proportion of shared
    features) of the top-k features across trials.
    Consistency is computed pairwise across trials.
    Input size is (n_trials, n_inputs, k)
    Returns a vector of size (n_inputs,)
    where each entry is the consistency of the
    top-k features across trials for that input."""

    # Initialize variables and arrays
    n_trials, n_inputs, k = top_k.shape
    consistency = np.zeros(n_inputs)

    # Compute consistency
    for input_idx in range(n_inputs):
        total_consistency = 0
        num_pairs = 0

        # Pairwise consistency
        for trial_a in range(n_trials):
            for trial_b in range(trial_a + 1, n_trials):
                tka = top_k[trial_a, input_idx]
                tkb = top_k[trial_b, input_idx]
                n_shared = len(np.intersect1d(tka, tkb))
                pair_consistency = n_shared / k
                total_consistency += pair_consistency
                num_pairs += 1

        consistency[input_idx] = total_consistency / num_pairs

    return consistency

def angle_diff(a, b, degrees=True):
    """
    Computes the angle between two vectors
    assumes a and b are normalized
    a and b have shape (n_inputs, n_features)
    angle has shape (n_inputs)
    """
    # Assumes a and b are normalized
    d = np.linalg.norm(a-b, axis=1)/2
    angle = np.arcsin(d)*2
    if degrees:
        angle *= 180/np.pi
    return angle

def average_pairwise_score_grad(grads, metric_func):
    """
    grads has shape (n_models, n_inputs, 2)
    and is not normalized
    angles has shape (n_inputs)
    and is the average angle between all
    pairs of models for each input
    """
    norm_grads = grads/np.linalg.norm(grads, axis=2)[:, :, np.newaxis]
    n_models, n_inputs, _ = grads.shape
    scores = np.zeros(n_inputs)
    for i in range(n_models):
        for j in range(i+1, n_models):
            scores += metric_func(norm_grads[i], norm_grads[j])

    total_pairs = n_models*(n_models-1)/2
    scores /= total_pairs
    
    return scores

def cosine_similarity(a, b):
    """
    Computes the cosine similarity between two vectors
    a and b have shape (n_inputs, n_features)
    assumes a and b are normalized
    """
    d = np.linalg.norm(a-b, axis=1)
    cosine = (1 - d**2)/2
    return cosine


_metrics_dict = {
    "sa": top_k_sa,
    "ssa": top_k_ssa,
    "cdc": top_k_cdc,
}

def _get_method_stats(idx):
    sims = np.zeros((len(ensemble_sizes), n_inputs_exp))
    for i, ensemble_size in enumerate(ensemble_sizes):
        if methods[idx] in ['mode connect', 'combined'] and ensemble_size == 1:
            sims[i] = np.nan
        else:
            sims[i] = average_pairwise_score(topk[idx, i], signs[idx, i], _metrics_dict[metric])
    return sims

def _get_similarities(ensemble_sizes, k, exp, metric, n_inputs_exp, directory, save=True,
                     methods=['average', 'majority', 'perturb', 'mode connect', 'combined']):

    with tqdm_joblib(tqdm(desc="Computing Statistics", total=len(methods))) as progress_bar:
        res = Parallel(n_jobs=num_cores)(delayed(_get_method_stats)(idx) for idx in range(len(methods)))

    similarities = np.zeros((len(methods), len(ensemble_sizes), n_inputs_exp))
    for i in range(len(methods)):
        similarities[i] = res[i]

    # Save similarities and test accuracies
    if save:
        np.save(f'{directory}/top{k}_{metric}_{exp}.npy', similarities)
        np.save(f'{directory}/ensemble_accs_{ensemble_sizes[0]}_to_{ensemble_sizes[-1]}.npy', test_accs)

    return similarities, test_accs

# Comparison of Ensemble Techniques
if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='heloc', help='dataset name')
    parser.add_argument('--explanation', type=str, default='', help='explanation name (gradients, smoothgrad, etc.)')
    parser.add_argument('--metric', type=str, default='sa', help='similarity metric')
    parser.add_argument('--k', type=int, default=5, help='top-k features to consider')
    parser.add_argument('--n_inputs_exp', type=int, default=1000, help='number of inputs to explain')
    parser.add_argument('--n_trials', type=int, default=50, help='number of ensembles to construct')
    parser.add_argument('--n_models', type=int, default=1000, help='number of models to sample from')

    # Get config dictionary
    args = vars(parser.parse_args())
    name = args['name']
    exp = args['explanation']
    metric = args['metric']
    k = args['k']
    n_inputs_exp = args['n_inputs_exp']
    n_trials = args['n_trials']
    n_models = args['n_models']

    # Ensemble sizes
    ensemble_sizes = [1, 2, 4, 6, 8, 12, 16, 20]
    random_source = 'rs'
    trainset, testset = datasets.load_dataset(name)

    X_test, y_test = testset.data.numpy(), testset.labels.numpy()
    n_inputs, n_features = X_test.shape
    n_inputs_exp = min(n_inputs_exp, n_inputs)
    model_args = [n_features, datasets.tabular.layers[name]]
    config = _load_config('train_params.json', name)
    optim = 'adam'
    epochs = config[optim]['epochs']
    lr = config[optim]['lr']
    batch_size = config[optim]['batch_size']
    dropout = config[optim]['dropout']
    directory = f'models/{name}/{random_source}/{optim}_epochs{epochs}_lr{lr}_batch{batch_size}_dropout{dropout}'
    print(bold("Directory:"), directory)
    print(bold("Number of features:"), n_features)
    print(bold("Ensemble sizes:"), ensemble_sizes)

    print(bold("Computing top-k statistics..."))
    methods=['average', 'majority', 'perturb', 'mode connect', 'combined']
    topk = np.zeros((len(methods), len(ensemble_sizes), n_trials, n_inputs_exp, k))
    signs = np.zeros((len(methods), len(ensemble_sizes), n_trials, n_inputs_exp, k), dtype=int)
    test_accs = np.zeros((len(methods), len(ensemble_sizes), n_trials))

    for e, ensemble_size in enumerate(tqdm(ensemble_sizes)):
        # Sample models
        model_idx = np.random.choice(n_models, (n_trials, ensemble_size), replace=False)
        for i in range(n_trials):
            for j, method in enumerate(methods):
                if method in ['mode connect', 'combined'] and ensemble_size == 1:
                    test_accs[j, e, i] = np.nan
                else:
                    grads, preds = get_statistics(model_idx[i], method, directory, exp=exp)
                    topk[j, e, i], signs[j, e, i] = get_top_k(k=k, X=grads, return_sign=True)
                    test_accs[j, e, i] = (preds == y_test).mean()

    print(bold("Running in parallel"))
    set_start_method('spawn', force=True)
    num_cores = cpu_count()
    print(bold("Number of cores:"), num_cores)
    similarities, test_accs = _get_similarities(ensemble_sizes, k, exp, metric, n_inputs_exp,
                                               directory, save=True, methods=methods)
    