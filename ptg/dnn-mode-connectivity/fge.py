import argparse
import numpy as np
import os
import sys
import tabulate
import time
import torch
from torch import nn
import torch.nn.functional as F

import utils

def fge(
    model: nn.Module,
    num_classes: int,
    trainloader,
    testset,
    testloader,
    epochs: int = 20,
    lr1: float = 0.05,
    lr2: float = 0.0001,
    wd: float = 1e-4,
    cycle: int = 4,
    momentum: float = 0.9,
    dir: float = "models/fge",
):
    assert cycle % 2 == 0, 'Cycle length should be even'

    os.makedirs(dir, exist_ok=True)
    with open(os.path.join(dir, 'fge.sh'), 'w') as f:
        f.write(' '.join(sys.argv))
        f.write('\n')

    criterion = F.cross_entropy

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr1,
        momentum=momentum,
        weight_decay=wd
    )

    ensemble_size = 0
    predictions_sum = np.zeros((len(testset), num_classes))

    columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_nll', 'te_acc', 'ens_acc', 'time']

    for epoch in range(epochs):
        time_ep = time.time()
        lr_schedule = utils.cyclic_learning_rate(epoch, cycle, lr1, lr2)
        train_res = utils.train(trainloader, model, optimizer, criterion, lr_schedule=lr_schedule)
        test_res = utils.test(testloader, model, criterion)
        time_ep = time.time() - time_ep
        predictions, targets = utils.predictions(testset, model)
        ens_acc = None
        if (epoch % cycle + 1) == cycle // 2:
            ensemble_size += 1
            predictions_sum += predictions
            ens_acc = 100.0 * np.mean(np.argmax(predictions_sum, axis=1) == targets)

        if (epoch + 1) % (cycle // 2) == 0:
            utils.save_checkpoint(
                dir,
                epoch,
                name='fge',
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict()
            )

        values = [epoch, lr_schedule(1.0), train_res['loss'], train_res['accuracy'], test_res['nll'],
                test_res['accuracy'], ens_acc, time_ep]
        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='9.4f')
        if epoch % 40 == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)
