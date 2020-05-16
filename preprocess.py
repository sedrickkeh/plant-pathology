import pandas as pd
import re
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold

import torch
from torch.utils.data import DataLoader

from dataset import PlantDataset

# def get_dataloaders(batch_size, n_folds):
#     df_train = pd.read_csv('./train.csv')
#     df_test = pd.read_csv('./test.csv')
#     df_train['class'] = np.argmax(df_train.iloc[:, 1:].values, axis=1)

#     # Stratified KFold Cross Validation
#     folds = StratifiedKFold(n_folds, shuffle = True, random_state = 644)
#     for i_fold, (train_idx, val_idx) in enumerate(folds.split(df_train, df_train['class'].values)):
#         df_train.loc[val_idx, 'fold'] = i_fold
#     df_train['fold'] = df_train['fold'].astype(int)

#     train_data = PlantDataset(df_train, "train")
#     train_size = int(0.8 * len(train_data))
#     val_size = len(train_data) - train_size
#     train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])
#     test_data = PlantDataset(df_test, "test")

#     data_loader_train = DataLoader(dataset=train_data,
#                                     batch_size=batch_size, 
#                                     shuffle=True)
#     data_loader_val = DataLoader(dataset=val_data,
#                                     batch_size=batch_size,
#                                     shuffle=False)
#     data_loader_test = DataLoader(dataset=test_data,
#                                     batch_size=batch_size, 
#                                     shuffle=False)

#     return data_loader_train, data_loader_val, data_loader_test


def get_data(n_folds):
    df_train = pd.read_csv('./train.csv')
    df_test = pd.read_csv('./test.csv')
    df_train['class'] = np.argmax(df_train.iloc[:, 1:].values, axis=1)

    # Stratified KFold Cross Validation
    folds = StratifiedKFold(n_folds, shuffle = True, random_state = 644)
    for i_fold, (train_idx, val_idx) in enumerate(folds.split(df_train, df_train['class'].values)):
        df_train.loc[val_idx, 'fold'] = i_fold
    df_train['fold'] = df_train['fold'].astype(int)
    return df_train, df_test


def get_train_val_dataloader(df_train, i_fold, batch_size):
    train_idx, val_idx = np.where((df_train['fold'] != i_fold))[0], np.where((df_train['fold'] == i_fold))[0]
    dataset_train = PlantDataset(df_train, train_idx, "train")
    dataset_val = PlantDataset(df_train, val_idx, "val")
    
    dataloader_train = DataLoader(dataset=dataset_train,
                                    batch_size=batch_size,
                                    shuffle=True)
    dataloader_val = DataLoader(dataset=dataset_val,
                                batch_size=batch_size,
                                shuffle=False)

    return dataloader_train, dataloader_val

def get_test_dataloader(df_test, batch_size):
    test_idx = np.array(range(len(df_test)))
    dataset_test = PlantDataset(df_test, test_idx, "test")
    dataloader_test = DataLoader(dataset=dataset_test,
                                    batch_size=batch_size, 
                                    shuffle=False)
    return dataloader_test


def edge_and_cut(img):
    emb_img = img.copy()
    edges = cv2.Canny(img, 100, 200)
    edge_coors = []
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            if edges[i][j] != 0:
                edge_coors.append((i, j))
    
    row_min = edge_coors[np.argsort([coor[0] for coor in edge_coors])[0]][0]
    row_max = edge_coors[np.argsort([coor[0] for coor in edge_coors])[-1]][0]
    col_min = edge_coors[np.argsort([coor[1] for coor in edge_coors])[0]][1]
    col_max = edge_coors[np.argsort([coor[1] for coor in edge_coors])[-1]][1]
    new_img = img[row_min:row_max, col_min:col_max]
    
    emb_img[row_min-10:row_min+10, col_min:col_max] = [255, 0, 0]
    emb_img[row_max-10:row_max+10, col_min:col_max] = [255, 0, 0]
    emb_img[row_min:row_max, col_min-10:col_min+10] = [255, 0, 0]
    emb_img[row_min:row_max, col_max-10:col_max+10] = [255, 0, 0]
    
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 20))
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('Original Image', fontsize=24)
    ax[1].imshow(edges, cmap='gray')
    ax[1].set_title('Canny Edges', fontsize=24)
    ax[2].imshow(emb_img, cmap='gray')
    ax[2].set_title('Bounding Box', fontsize=24)
    plt.show()

