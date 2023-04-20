"""FashionMNIST data, pipeline, and model. TODO: rewrite fmnist saving structure?"""

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb
from tqdm import tqdm
from util import State, convert_to_tensor
from typing import Tuple
import math
from datasets import curves


def load_fmnist_dataset() -> Tuple[datasets.FashionMNIST, datasets.FashionMNIST]:
    """
    Load FashionMNIST dataset, return train and test sets
    Downloads if not in default root directory of ./data/fmnist/
    Transform is torchvision.transforms.ToTensor()
    """

    # Load trainset
    trainset = datasets.FashionMNIST(
        root='../data/fmnist/',
        download=True,
        train=True,
        transform=transforms.ToTensor(),
    )

    # Load testset
    testset = datasets.FashionMNIST(
        root='../data/fmnist/',
        download=True,
        train=False,
        transform=transforms.ToTensor(),
    )

    # Return train/test sets
    return trainset, testset


def learning_pipeline(S: State) -> nn.Module:
    """Learning pipeline for FashionMNIST dataset"""

    # Random seed (controls initialization and SGD stochasticity)
    torch.manual_seed(S.seed)
    
    # Initialize model and set to train mode
    S.net = S.net(num_classes=10, dropout=S.config['dropout'])
    S.net.train()

    # Set up loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        S.net.parameters(),
        lr=S.config['lr'],
    )

    # Set up dataloader
    loader = DataLoader(
        S.trainset,
        batch_size=S.config['batch_size'],
    )

    wandb.watch(S.net, loss_fn, log='all', log_freq=1)
    epochs = S.config['epochs']

    with tqdm(total=epochs*len(loader), desc=wandb.run.name) as pbar:
        for _ in range(epochs):
            for i, (x, y) in enumerate(loader):
                optimizer.zero_grad()
                y_pred = S.net(x)
                loss = loss_fn(y_pred, y)
                loss.backward()
                optimizer.step()
                if i % 100 == 0:
                    acc = (y_pred.argmax(-1) == y).float().mean()
                    # Log metrics to wandb
                    wandb.log({
                        'loss': loss,
                        'acc': acc,
                    })
                pbar.update(1)

    # Return the trained model
    return S.net


class LeNet5(nn.Module):
    """LeNet5 model class for FashionMNIST dataset"""

    def __init__(self, num_classes: int, dropout: float):
        """Initialize the model"""

        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """Forward pass method, returns softmax predictions
        x must be a tensor of shape (n, 1, 28, 28)"""

        z = self.layer1(x)
        z = self.layer2(z)
        z = z.flatten(start_dim=1)
        z = self.dropout(z)
        z = self.relu(self.fc1(z))
        z = self.dropout(z)
        z = self.relu(self.fc2(z))
        z = self.dropout(z)
        z = self.softmax(self.fc3(z))

        # Return softmax predictions
        return z

    def predict(self, x, return_numpy=False):
        """Predict method, returns hard predictions
        Flexible to take in tensor or numpy array
        Shape of x is (n, 1, 28, 28) or (n, 28, 28) or (28, 28)
        Returns numpy array if return_numpy is True"""

        # Convert input to tensor if it's a numpy array
        x = convert_to_tensor(x)

        # Add extra dimension if input is size (28, 28)
        x = x.unsqueeze(0) if len(x.shape) == 2 else x  # size 2 -> 3

        # Add extra dimension if input is of size (n, 28, 28)
        x = x.unsqueeze(1) if len(x.shape) == 3 else x  # size 3 -> 4

        # Forward pass and argmax for hard prediction
        self.eval()
        with torch.no_grad():  # save memory (inference only)
            preds = self(x.float()).argmax(dim=1)  # softmax output

        # Return hard predictions
        if return_numpy:
            return preds.detach().numpy()
        return preds
    


class LeNet5Curve(nn.Module):
    """LeNet5 curve model class for FashionMNIST dataset"""

    def __init__(self, num_classes: int, fix_points: list[bool], dropout: float = 0.0):
        super(LeNet5Curve, self).__init__()
        self.conv1 = curves.Conv2d(1, 6, kernel_size=5, stride=1, padding=2, fix_points=fix_points)
        self.batch1 = curves.BatchNorm2d(6, fix_points=fix_points)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = curves.Conv2d(6, 16, kernel_size=5, stride=1, padding=0, fix_points=fix_points)
        self.batch2 = curves.BatchNorm2d(16, fix_points=fix_points)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = curves.Linear(400, 120, fix_points=fix_points)
        self.fc2 = curves.Linear(120, 84, fix_points=fix_points)
        self.fc3 = curves.Linear(84, num_classes, fix_points=fix_points)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, curves.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.num_bends):
                    getattr(m, 'weight_%d' % i).data.normal_(0, math.sqrt(2. / n))
                    getattr(m, 'bias_%d' % i).data.zero_()

    def forward(self, x, coeffs_t):
        """Forward pass method, returns softmax predictions
        x must be a tensor of shape (n, 1, 28, 28)"""

        z = self.conv1(x, coeffs_t)
        z = self.batch1(z, coeffs_t)
        z = self.relu(z)
        z = self.max_pool1(z)

        z = self.conv2(z, coeffs_t)
        z = self.batch2(z, coeffs_t)
        z = self.relu(z)
        z = self.max_pool2(z)

        z = z.flatten(start_dim=1)
        z = self.dropout(z)
        z = self.relu(self.fc1(z, coeffs_t))
        z = self.dropout(z)
        z = self.relu(self.fc2(z, coeffs_t))
        z = self.dropout(z)
        z = self.softmax(self.fc3(z, coeffs_t))

        # Return softmax predictions
        return z


# def state_sampler() -> State:
#     """Sample a state for the learning pipeline
#     We need to fix this. trainset should be included and training seed specified
#     """
#     trainset, _ = load_fmnist_dataset()

#     if wandb.config['loo']:
#         torch.manual_seed(0)  # Set seed for reproducibility
#         mask = torch.randperm(len(trainset))
#         trainset = Subset(trainset, mask[:int(0.9 * len(mask))])
#         torch.manual_seed(0)

#     # Return a state object
#     return State(
#         LeNet5(num_classes=10, dropout=wandb.config['dropout']),
#         trainset,
#         wandb.config,
#     )