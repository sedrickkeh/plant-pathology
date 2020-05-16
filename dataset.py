import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class PlantDataset(Dataset):
    def __init__(self, df, idx, mode, transforms=None):
        self.df = df
        self.idx = np.asarray(idx).astype('int')
        self.transforms=transforms
        self.mode = mode
        
    def __len__(self):
        return self.idx.shape[0]
    
    def __getitem__(self, index):
        idx = self.idx[index]
        image_src = './images/' + self.df.loc[idx, 'image_id'] + '.jpg'
        # print(image_src)
        image = cv2.imread(image_src, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 320), interpolation = cv2.INTER_AREA).astype('uint8')
        image = np.transpose(image, (2, 0, 1))
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed['image']

        if (self.mode == "test"):
            return image, self.df.loc[idx, 'image_id']
        
        # labels = self.df.loc[idx, ['healthy', 'multiple_diseases', 'rust', 'scab']].values
        # labels = torch.from_numpy(labels.astype(np.int8))
        # labels = labels.unsqueeze(-1)
        labels = self.df.loc[idx, ['class']].values
        labels = torch.from_numpy(labels.astype(np.int8))
        labels = labels.squeeze(-1)
        labels = labels.long()
        return image, labels