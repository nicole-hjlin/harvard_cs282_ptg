import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Subset
import wandb
from tqdm import tqdm
from util import State
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Hyperparameters and architecture used in Black et al. (500 models)
# They have 61 features, we have 70 (zero mean, unit variance)
# epochs = 100
# batch_size = 32
# dropout = 0
# lr = unspecified


# 20 features in the dataset
cols = ['status', 'duration', 'credit_history', 'purpose', 'amount', 'savings', 'employment_duration',
        'installment_rate', 'personal_status_sex', 'other_debtors', 'present_residence', 'property', 'age',
        'other_installment_plans', 'housing', 'number_credits', 'job', 'people_liable', 'telephone', 'foreign_worker', 'label']

cats = ['status', 'credit_history', 'purpose', 'savings', 'employment_duration',
        'installment_rate', 'personal_status_sex', 'other_debtors', 'present_residence',
        'property', 'other_installment_plans', 'housing', 'number_credits', 'job',
        'people_liable', 'telephone', 'foreign_worker']

# Preprocess the German dataset
def preprocess_german(df: pd.DataFrame) -> torch.Tensor:
    """Preprocess the German dataset
    Takes in a dataframe and returns a processed tensor ()
    """

    # Set column names
    df.columns = cols

    # Convert the label to 0 and 1
    df['label'] = 2-df['label']

    # Convert the categorical columns to one-hot
    df = pd.get_dummies(df, columns=cats)
    
    y = df['label']
    columns = df.columns[:-1]
    x = df.drop('label', axis=1)

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    data = pd.DataFrame(x, columns=columns)
    data['label'] = y

    return data