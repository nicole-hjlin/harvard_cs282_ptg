from util import State
from torch import optim, nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import wandb

trainset = datasets.FashionMNIST(
    'MNIST_data/',
    download=True,
    train=True,
    transform=transforms.ToTensor(),
)

def state_sampler() -> State:
    return State(
        LeNet5(num_classes=10, dropout=0.1),
        trainset,
        wandb.config,
    )

def learning_pipeline(S: State) -> nn.Module:
    S.net.train()
    wandb.watch(S.net)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        S.net.parameters(),
        lr=S.hyperparameters['lr'],
    )
    loader = DataLoader(
        S.trainset,
        batch_size=S.hyperparameters['batch_size'],
    )

    for _ in range(S.hyperparameters['epochs']):
        for i, (x, y) in enumerate(loader):
            optimizer.zero_grad()
            y_pred = S.net(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                wandb.log({'loss': loss})

    return S.net

class LeNet5(nn.Module):
    def __init__(self, num_classes: int, dropout: float):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(256, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        z = self.layer1(x)
        z = self.layer2(z)
        z = z.flatten(start_dim=1)
        z = self.dropout(z)
        z = self.fc(z)
        z = self.relu(z)
        z = self.fc1(z)
        z = self.relu1(z)
        z = self.dropout(z)
        z = self.fc2(z)
        return z
