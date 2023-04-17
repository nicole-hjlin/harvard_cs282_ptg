import numpy as np
import torch
from tqdm import tqdm
from train import get_model_class

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

def top_k_ssa(k, x, y, signs_x, signs_y):
    ''' Returns Signed Set *Agreement* (i.e., 1 means perfect agreement)'''
    # First, we need to satisfy CDC, so check that first: 
    limited_sx = signs_x[np.arange(x.shape[0])[:,None], x]
    limited_sy = signs_y[np.arange(x.shape[0])[:,None], x]
    xeq = (limited_sx == limited_sy)#  + (-1) * (limited_sx != limited_sy)
    limited_sx = signs_x[np.arange(y.shape[0])[:,None], y]
    limited_sy = signs_y[np.arange(y.shape[0])[:,None], y]
    yeq = (limited_sx == limited_sy)
    cdc = np.logical_and(np.all(xeq == 1, axis=1), np.all(yeq == 1, axis=1))

    # Next, we need to know whether X and Y have the same top-k features
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
    scores = np.logical_and(frac_right, cdc)
    return sum(scores)/len(scores)

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


def top_k_experiment(ensemble_sizes, n_trials, test_idx, k, x,
                     n_models, name, model_args, random_source):
    # Initialize
    directory = f'models/{name}'
    model_class = get_model_class(name)
    n_features = x.shape[1]

    # Count variables
    topk, signs, counts = [], [], []
    counts_pos, counts_neg = [], []
    topk_counts_signed, signs_counts_signed = [], []

    # 'Average ensemble' variables (take mean of gradients across models)
    topk_avg = np.zeros((len(ensemble_sizes), n_trials, len(test_idx), k))
    signs_avg = np.zeros((len(ensemble_sizes), n_trials, len(test_idx), k), dtype=int)

    # 'Norm-average ensemble' variables (normalize gradients to unit l2-norm for each model, then take mean)
    topk_norm_avg = np.zeros((len(ensemble_sizes), n_trials, len(test_idx), k))
    signs_norm_avg = np.zeros((len(ensemble_sizes), n_trials, len(test_idx), k), dtype=int)

    # 'Average smooth ensemble' variables (for each model, add noise to weights and recompute gradients, take mean)
    topk_smooth = np.zeros((len(ensemble_sizes), n_trials, len(test_idx), k))
    signs_smooth = np.zeros((len(ensemble_sizes), n_trials, len(test_idx), k), dtype=int)
    n_weight_perturbations = 20
    sigma = 0.5
    layer_str = 'network.0.weight'

    for e, ensemble_size in tqdm(enumerate(ensemble_sizes)):
        tk = np.zeros((n_trials, ensemble_size, len(test_idx), k))
        s = np.zeros((n_trials, ensemble_size, len(test_idx), k), dtype=int)
        co = np.zeros((n_trials, len(test_idx), n_features))
        
        # Initialize count arrays
        co_pos = np.zeros((n_trials, len(test_idx), n_features))
        co_neg = np.zeros((n_trials, len(test_idx), n_features))

        for i in range(n_trials):
            model_idx = np.random.choice(n_models, ensemble_size, replace=False)
            grads = np.array([np.load(f'{directory}/{random_source}_grads_{idx}.npy') for idx in model_idx])
            grads = grads[:, test_idx]
            tk[i], s[i] = get_top_k(k=k, X=grads, return_sign=True)
            topk_avg[e, i], signs_avg[e, i] = get_top_k(k=k, X=grads.mean(axis=0), return_sign=True)
            norm_grads = grads/np.linalg.norm(grads+1e-20, axis=2, keepdims=True)
            topk_norm_avg[e, i], signs_norm_avg[e, i] =\
                get_top_k(k=k, X=norm_grads.mean(axis=0), return_sign=True)
            
            # Smooth ensemble
            smooth_grads = np.zeros_like(grads)
            for j in range(ensemble_size):
                for _ in range(n_weight_perturbations):
                    # Add noise to layer weights
                    model = model_class(*model_args)
                    state_dict = torch.load(f'{directory}/{random_source}_model_{model_idx[j]}.pth')
                    state_dict[layer_str] += torch.randn(state_dict[layer_str].shape) * sigma
                    model.load_state_dict(state_dict)

                    # Compute new gradients
                    smooth_grads[j] += model.compute_gradients(x, softmax=False, label=1, return_numpy=True)
            smooth_grads /= n_weight_perturbations

            topk_smooth[e, i], signs_smooth[e, i] =\
                get_top_k(k=k, X=smooth_grads.mean(axis=0), return_sign=True)


            # Compute counts
            for t_i in range(len(test_idx)):
                # Unique indices and their counts
                u, c = np.unique(tk[i, :, t_i].flatten(), return_counts=True)
                co[i, t_i, u.astype(int)] = c
                for j in range(ensemble_size):
                    for l in range(k):
                        feature_idx = tk[i, j, t_i, l].astype(int)
                        feature_sign = s[i, j, t_i, l]

                        if feature_sign > 0:
                            co_pos[i, t_i, feature_idx] += 1
                        elif feature_sign < 0:
                            co_neg[i, t_i, feature_idx] += 1
        tk_co_s, s_co_s = get_top_k(k, co_pos-co_neg, return_sign=True)
        
        # Append to lists
        topk.append(tk)
        signs.append(s)
        counts.append(co)
        counts_pos.append(co_pos)
        counts_neg.append(co_neg)
        topk_counts_signed.append(tk_co_s)
        signs_counts_signed.append(s_co_s)

    return topk_counts_signed, signs_counts_signed, topk_avg, signs_avg, topk_norm_avg, signs_norm_avg, topk_smooth, signs_smooth