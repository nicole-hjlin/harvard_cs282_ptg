import torch
import torch.nn.functional as F

from modconn import curves
from modconn import utils

from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm


def learning_rate_schedule(base_lr, epoch, total_epochs):
    alpha = epoch / total_epochs
    if alpha <= 0.5:
        factor = 1.0
    elif alpha <= 0.9:
        factor = 1.0 - (alpha - 0.5) / 0.4 * 0.99
    else:
        factor = 0.01
    return factor * base_lr


def custom_cross_entropy_loss(output, target, target_loss):
    ce_loss = F.cross_entropy(output, target)
    loss_diff = (ce_loss - target_loss).abs()
    return loss_diff


def train_curve(
    models: list,
    trainloader: Dataset,
    num_classes: int,
    model_class: nn.Module,
    curve_class: nn.Module,
    curve: nn.Module,
    # num_bends: int,
    # init_start: str,
    # init_end: str,
    fix_start: bool,
    fix_end: bool, #first_layer: bool,
    model_args,
    lr: float,
    momentum: float,
    epochs: int,
    disable_tqdm: bool = True,
    init: str = 'linear',
    target_loss: float = 0.5,
):
    # Create model (with initial parameters)
    model = curves.CurveNet(
        num_classes,
        curve,  # polychain
        curve_class,  # architecture
        len(models),  # num_bends
        model_args,
        fix_start,
        fix_end,
    )
    #base_model = None
    # Load initial parameters
    for i, base_model in enumerate(models):
        #if path is not None:
        # if base_model is None:
        #     base_model = model_class(input_size=23,
        #                                 hidden_layers=[128,64,16])
        # state_dict = torch.load(path)
        # base_model.load_state_dict(state_dict)
        if base_model is not None:
            model.import_base_parameters(base_model, i)
    if init == 'linear':
        model.init_linear()
    elif init == 'radial':
        model.init_radial()
    elif init =='none':
        pass
    else:
        raise ValueError('Invalid init: %s' % init)

    #criterion = F.cross_entropy
    criterion = custom_cross_entropy_loss
    target_loss = torch.tensor(target_loss)
    optimizer = torch.optim.Adam(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=lr,
        #momentum=momentum,
    )

    for epoch in tqdm(range(epochs), disable=disable_tqdm):
        #lr = learning_rate_schedule(lr, epoch, epochs)
        #utils.adjust_learning_rate(optimizer, lr)

        utils.train(trainloader, model, optimizer, criterion, target_loss)

    return model
