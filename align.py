import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import copy
import torch.nn as nn
from datasets.tabular import TabularModel, TabularModelAlign

def convert_model(new_model, model_args, to_align=True):
    new_state_dict = new_model.state_dict()
    new_state_dict_keys = list(new_state_dict.keys())
    model_class = TabularModelAlign if to_align else TabularModel
    model = model_class(*model_args)
    state_dict = model.state_dict()
    for i, key in enumerate(state_dict.keys()):
        state_dict[key] = new_state_dict[new_state_dict_keys[i]]
    model.load_state_dict(state_dict)
    return model

def align_tabular(model0, model1, trainloader, model_args):
    # Convert models
    model_0 = convert_model(model0, model_args)
    model_1 = convert_model(model1, model_args)
    hard_match, _, _, _ = compute_model_alignment(model_1, model_0, trainloader)
    hard_match = [i[:, 1] for i in hard_match]
    model_a = align_models(model_0, hard_match)
    model_a = convert_model(model_a, model_args, to_align=False)
    return model_a

def get_activations(test_loader, model, idx_layer,
                    acts_old=None, idx_warmstart=None,
                    review=True, use_warmstart=True):
    acts = []
    model.eval()
    if acts_old is None or not use_warmstart:
        for input, _ in test_loader:
            act_new = model.get_activation(input, idx_act=idx_layer)
            acts += [act_new]
    else:
        for act_old in acts_old:
            act_new = model.get_activation(act_old, idx_act=idx_layer, idx_warmstart=idx_warmstart)
            acts += [act_new]
    act = torch.cat([torch.transpose(i, 0, 1) for i in acts], dim=1)
    if review:
        act = torch.flatten(act, start_dim=1)
    act = act.cpu().detach().numpy()
    return act, acts

def compute_alignment(corr):
    num_filter = corr.shape[0]
    hard_match = np.zeros([num_filter, 2])
    hard_match[:, 0], hard_match[:, 1] = linear_sum_assignment(1.01 - corr)
    hard_match = hard_match.astype(int)
    return hard_match

def get_zscore(model, dataloader, idx_layer, acts_old=None, idx_warmstart=None, use_warmstart=True):
    eps = 1E-5
    act, acts = get_activations(dataloader, model, idx_layer, acts_old=acts_old,
                                idx_warmstart=idx_warmstart, use_warmstart=use_warmstart)
    act_mu = np.expand_dims(np.mean(act, axis=1), -1)
    act_sigma = np.expand_dims(np.std(act, axis=1), -1) + eps  # Avoid divide by zero
    z_score = (act - act_mu) / act_sigma
    return z_score, acts

def compute_alignment_random(corr, seed=None):
    num_filter = corr.shape[0]
    random_match = np.zeros([num_filter, 2])
    random_match[:, 0] = range(num_filter)
    random_match[:, 1] = np.random.RandomState(seed=seed).permutation(num_filter)
    return random_match

def compute_corr(model0, model1, dataloader, idx_layer, acts_old0=None, acts_old1=None, idx_warmstart=None,
                 use_warmstart=True):
    z_score0, acts0 = get_zscore(model0, dataloader, idx_layer, acts_old=acts_old0,
                                 idx_warmstart=idx_warmstart, use_warmstart=use_warmstart)
    z_score1, acts1 = get_zscore(model1, dataloader, idx_layer, acts_old=acts_old1,
                                 idx_warmstart=idx_warmstart, use_warmstart=use_warmstart)
    corr = np.matmul(z_score0, z_score1.transpose()) / z_score0.shape[1]
    return corr, acts0, acts1

def compute_model_alignment(model0, model1, dataloader, num_layer=None,
                            return_corr=None, quad_assignment=False,
                            use_warmstart=True):
    if num_layer is None:
        num_layer = len(model0.layers) + 1

    hard_match = [None] * (num_layer - 1)
    random_match = [None] * (num_layer - 1)

    corr_unaligned_mean = np.zeros(num_layer)
    corr_aligned_mean = np.zeros(num_layer)

    if return_corr is not None:
        corr_unaligned_returned = []
        corr_aligned_returned = []
    else:
        corr_unaligned_returned = []
        corr_aligned_returned = []

    acts_old0 = None
    acts_old1 = None
    for layer in range(num_layer):
        #print('Layer %d' % layer)

        if not quad_assignment:
            if use_warmstart:
                corr, acts_old0, acts_old1 = compute_corr(model0, model1, dataloader, layer, acts_old0, acts_old1,
                                                            idx_warmstart=layer - 1)
                # print(acts_old0[0].shape)
                # print(acts_old0.shape, acts_old1.shape)
                # print(acts_old0, acts_old1)
            else:
                corr, _, _ = compute_corr(model0, model1, dataloader, layer, idx_warmstart=None, use_warmstart=False)
            if layer < num_layer - 1:
                hard_match[layer] = compute_alignment(corr)
                random_match[layer] = compute_alignment_random(corr, seed=layer)
                corr_aligned = corr[:, hard_match[layer][:, 1]]
            else:
                corr_aligned = corr
            corr_unaligned_mean[layer] = np.mean(np.diag(corr))
            corr_aligned_mean[layer] = np.mean(np.diag(corr_aligned))

            if return_corr is not None:
                if layer in return_corr:
                    corr_unaligned_returned.append(corr)
                    corr_aligned_returned.append(corr_aligned)
                if layer >= np.max(return_corr):
                    return hard_match, random_match, corr_unaligned_mean, corr_aligned_mean, \
                        corr_unaligned_returned, corr_aligned_returned

    return hard_match, random_match, corr_unaligned_mean, corr_aligned_mean

def align_models(model, matching, selected_layers=None):
    if selected_layers is None:
        selected_layers = np.arange(len(model.network) + 1)
    model_new = copy.deepcopy(model)
    for i, block in enumerate(model_new.network):
        if i in selected_layers:
            for layer in block.modules():
                align_weights_head(layer, matching[i])
                if i > 0:
                    align_weights_tail(layer, matching[i-1])
    if len(model.network) in selected_layers:
        if hasattr(model_new, 'dense'):  # TinyTen classifier
            pad_int = int(model_new.dense.in_features / matching[-1].size)
            new_match = []
            for mat in matching[-1]:
                new_match += [mat * pad_int + m_idx for m_idx in range(pad_int)]
            align_weights_tail(model_new.dense, new_match)
        if hasattr(model_new, 'classifier'):
            align_weights_tail(model_new.classifier, matching[-1])
    return model_new

def align_weights_head(layer, match):
    match = np.array(match, dtype=np.int) 
    if match.ndim == 1:
        if isinstance(layer, nn.Linear):
            layer.weight.data = layer.weight.data[match]
            if layer.bias is not None:
                layer.bias.data = layer.bias.data[match]
    else:
        assert match.ndim == 2
        if isinstance(layer, nn.Linear):
            layer.weight.data = torch.matmul(torch.tensor(match, device=layer.weight.device), layer.weight.data)
            if layer.bias is not None:
                layer.bias.data = torch.matmul(torch.tensor(match, device=layer.bias.device), layer.bias.data)

def align_weights_tail(layer, match):
    match = np.array(match, dtype=np.int) 
    if match.ndim == 1:
        if isinstance(layer, nn.Linear):
            layer.weight.data = layer.weight.data[:, match]
    else:
        assert match.ndim == 2
        if isinstance(layer, nn.Linear):
            match_t = torch.tensor(match, device=layer.weight.device)
            layer.weight.data = torch.matmul(layer.weight.data, match_t.t())