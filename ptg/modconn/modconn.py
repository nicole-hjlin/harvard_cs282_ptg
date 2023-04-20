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


def train_curve(
    trainloader: Dataset,
    num_classes: int,
    model_class: nn.Module,
    curve_class: nn.Module,
    curve: nn.Module,
    num_bends: int,
    init_start: dict,
    init_end: dict,
    fix_start: bool,
    fix_end: bool,
    model_args,
    lr: float,
    momentum: float,
    epochs: int,
):
    model = curves.CurveNet(
        num_classes,
        curve,  #polychain
        curve_class,  #architecture
        num_bends,
        model_args,
        fix_start,
        fix_end,
    )
    base_model = None
    for path, k in [(init_start, 0), (init_end, num_bends - 1)]:
        if path is not None:
            if base_model is None:
                base_model = model_class(input_size=23,
                                         hidden_layers=[128,64,16])
            state_dict = torch.load(path)
            base_model.load_state_dict(state_dict)
            model.import_base_parameters(base_model, k)
    model.init_linear()

    criterion = F.cross_entropy
    optimizer = torch.optim.SGD(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=lr,
        momentum=momentum,
    )

    for epoch in tqdm(range(epochs)):
        lr = learning_rate_schedule(lr, epoch, epochs)
        utils.adjust_learning_rate(optimizer, lr)

        utils.train(trainloader, model, optimizer, criterion)

    return model
