import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Subset
import wandb
from tqdm import tqdm
from util import State
import pandas as pd
from dataset import CSVDataset


def preprocessAdultSpecifically(df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.from_numpy(df.values).float()
    x = torch.zero_like(x)
    y = x[:,-1]
    x = x[:,:-1]
    return x, y

dataset = CSVDataset(
    source_url="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    download_path="../data/adult/adult.data",
    sep=" +",
    preprocess=preprocessAdultSpecifically,
)

# dataset, testset = split(dataset)
testset = CSVDataset(
    source_url="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    download_path="../data/adult/adult.data",
    sep=" +",
    preprocess=preprocessAdultSpecifically,
)

def state_sampler() -> State:
    """Sample a state for the learning pipeline"""
    if wandb.config['loo']:
        torch.seed()
        mask = torch.randperm(len(dataset))
        trainset = Subset(dataset, mask[:int(0.9 * len(mask))])
        torch.manual_seed(0)
    else:
        trainset = dataset

    # Return a state object
    return State(
        LeNet5(num_classes=10, dropout=wandb.config['dropout']),
        trainset,
        wandb.config,
    )


def learning_pipeline(S: State) -> nn.Module:
    """Learning pipeline for FashionMNIST dataset"""

    # Set up training
    S.net.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        S.net.parameters(),
        lr=S.hyperparameters['lr'],
    )
    loader = DataLoader(
        S.trainset,
        batch_size=S.hyperparameters['batch_size'],
    )

    wandb.watch(S.net, loss_fn, log='all', log_freq=1)
    epochs = S.hyperparameters['epochs']

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
