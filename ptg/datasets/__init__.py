"""Dataset module to load (and download) datasets. Reminder to put this in a package for AI4LIFE."""
from .fmnist import load_fmnist_dataset
from .tabular import load_tabular_dataset

def load_dataset(name):
    if name == 'fmnist':
        return load_fmnist_dataset()
    elif name in ['german', 'adult', 'heloc']:
        return load_tabular_dataset(name)
    else:
        raise ValueError(f"Unknown dataset: {name}")


# class TabularDataset(Dataset):
#     def __init__(self,
#         source_url: str,
#         download_path: str,
#         sep: str,
#         preprocess: Callable[[pd.DataFrame], Tuple[torch.Tensor, torch.Tensor]],
#     ):
#         # Create the directory if it doesn't exist
#         makedirs(path.dirname(download_path), exist_ok=True)
        
#         # Check if the file doesn't exist, then download and save it
#         if not path.isfile(download_path):
#             with open(download_path, 'wb') as f:
#                 r = requests.get(source_url, allow_redirects=True)
#                 f.write(r.content)
#             source_url = download_path

#         df = pd.read_csv(
#             source_url,
#             sep=sep,
#             engine="python",
#         )

#         x, y = preprocess(df)

#         self.x = x
#         self.y = y

#     def __len__(self):
#         return len(self.x)

#     def __getitem__(self, idx):
#         return self.x[idx], self.y[idx]
