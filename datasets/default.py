import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 23 features in dataset
cols = ['Limit_Bal', 'Sex', 'Education', 'Marriage', 'Age', 'Pay_0',
        'Pay_2', 'Pay_3', 'Pay_4', 'Pay_5', 'Pay_6', 'Bill_Amt1',
        'Bill_Amt2', 'Bill_Amt3', 'Bill_Amt4', 'Bill_Amt5',
        'Bill_Amt6', 'Pay_Amt1', 'Pay_Amt2', 'Pay_Amt3', 'Pay_Amt4',
        'Pay_Amt5', 'Pay_Amt6', 'label']

cats = ['Sex', 'Education', 'Marriage', 'Pay_0', 'Pay_2',
        'Pay_3', 'Pay_4', 'Pay_5', 'Pay_6']

# Preprocess the German dataset
def preprocess_default(df: pd.DataFrame) -> torch.Tensor:
    """Preprocess the German dataset
    Takes in a dataframe and returns a processed tensor ()
    """
    # Drop the ID column
    df = df.drop('ID', axis=1)

    # Set column names
    df.columns = cols

    # Convert the label to 0 and 1
    df['label'] = 1-df['label']

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
