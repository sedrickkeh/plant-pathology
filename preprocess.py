import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader

from dataset import PlantDataset

def get_dataloaders(batch_size):
    df_train = pd.read_csv('./train.csv')
    df_test = pd.read_csv('./test.csv')
    df_train['class'] = np.argmax(df_train.iloc[:, 1:].values, axis=1)
    
    labels_train = df_train.loc[:, 'healthy':'scab']
    df_train, df_val = train_test_split(df_train, test_size = 0.2, 
                                            random_state=23, stratify=labels_train)
    
    df_train.reset_index(inplace=True) 
    df_val.reset_index(inplace=True)

    train_data = PlantDataset(df_train, "train")
    val_data = PlantDataset(df_val, "val")
    test_data = PlantDataset(df_test, "test")

    data_loader_train = DataLoader(dataset=train_data,
                                    batch_size=batch_size, 
                                    shuffle=True)
    data_loader_val = DataLoader(dataset=val_data,
                                    batch_size=batch_size,
                                    shuffle=True)
    data_loader_test = DataLoader(dataset=test_data,
                                    batch_size=batch_size, 
                                    shuffle=False)

    return data_loader_train, data_loader_val, data_loader_test
