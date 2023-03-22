import requests
from io import StringIO
from os import makedirs, path
import torch
import pandas as pd
from torch.utils.data import Dataset
from typing import Tuple, Callable

def download_csv(url):
    response = requests.get(url)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))

class CSVDataset(Dataset):
    def __init__(self,
        source_url: str,
        download_path: str,
        sep: str,
        preprocess: Callable[[pd.DataFrame], Tuple[torch.Tensor, torch.Tensor]],
    ):
        if not path.exists(download_path):
            makedirs(path.dirname(download_path))
            with open(download_path, 'wb') as f:
                r = requests.get(source_url, allow_redirects=True)
                f.write(r.content)
            source_url = download_path

        df = pd.read_csv(
            source_url,
            sep=sep,
            engine="python",
        )

        x, y = preprocess(df)

        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
