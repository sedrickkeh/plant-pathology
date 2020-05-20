import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class EfficientNetModel(nn.Module):
    def __init__(self, args):
        super(EfficientNetModel, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b5')
        end_features = self.model._fc.in_features
        self.model._fc = nn.Sequential(nn.Linear(end_features,1000,bias=True),
                                nn.ReLU(),
                                nn.Dropout(p=0.5),
                                nn.Linear(1000,4, bias = True))
    
    def forward(self, x):
        return self.model(x)    

