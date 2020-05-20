import pandas as pd
import numpy as np
import torch
import torch.utils.data as Data
from sklearn.model_selection import train_test_split

from utils import *
from dataset import PlantDataset


def get_data():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    train['image_path'] = train['image_id'].apply(get_image_path)
    test['image_path'] = test['image_id'].apply(get_image_path)
    train_labels = train.loc[:, 'healthy':'scab']
    train_paths = train.image_path
    test_paths = test.image_path

    train_paths, valid_paths, train_labels, valid_labels = train_test_split(train_paths, train_labels, test_size = 0.2, random_state=23, stratify = train_labels)
    train_paths.reset_index(drop=True,inplace=True)
    train_labels.reset_index(drop=True,inplace=True)
    valid_paths.reset_index(drop=True,inplace=True)
    valid_labels.reset_index(drop=True,inplace=True)
    return train_paths, valid_paths, test_paths, train_labels, valid_labels


def get_dataloaders(train_paths, valid_paths, test_paths, train_labels, valid_labels, bsz):
    train_dataset = PlantDataset(train_paths, train_labels)
    trainloader = Data.DataLoader(train_dataset, shuffle=True, batch_size = bsz, num_workers = 2)

    valid_dataset = PlantDataset(valid_paths, valid_labels, train = False)
    validloader = Data.DataLoader(valid_dataset, shuffle=False, batch_size = bsz, num_workers = 2)

    test_dataset = PlantDataset(test_paths, train = False, test = True)
    testloader = Data.DataLoader(test_dataset, shuffle=False, batch_size = bsz, num_workers = 2)

    return trainloader, validloader, testloader