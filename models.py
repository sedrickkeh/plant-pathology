import torch
import torch.nn as nn
from torchvision import models
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

class EfficientNetModel_NoisyStudent(nn.Module):
    def __init__(self, args):
        super(EfficientNetModel_NoisyStudent, self).__init__()
        self.model = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'tf_efficientnet_b5_ns', pretrained=True)
        end_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(nn.Linear(end_features,1000,bias=True),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(1000,4, bias = True))
    def forward(self, x):
        return self.model(x)   



class DenseNetModel(nn.Module):
    def __init__(self, args):
        super(DenseNetModel, self).__init__()
        self.model = models.densenet161(pretrained=True)
        end_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(nn.Linear(end_features,1000,bias=True),
                                            nn.ReLU(),
                                            nn.Dropout(p=0.5),
                                            nn.Linear(1000,4, bias = True))
    def forward(self, x):
        return self.model(x)   



class ResNetModel(nn.Module):
    def __init__(self, args):
        super(ResNetModel, self).__init__()
        self.model = models.resnet101(pretrained=True)
        end_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Linear(end_features,1000,bias=True),
                                            nn.ReLU(),
                                            nn.Dropout(p=0.5),
                                            nn.Linear(1000,4, bias = True))
    def forward(self, x):
        return self.model(x)   



class VGGNetModel(nn.Module):
    def __init__(self, args):
        super(VGGNetModel, self).__init__()
        self.model = models.vgg16_bn(pretrained=True)
        end_features = 512 * 7 * 7
        self.model.classifier = nn.Sequential(nn.Linear(end_features,1000,bias=True),
                                            nn.ReLU(),
                                            nn.Dropout(p=0.5),
                                            nn.Linear(1000,4, bias = True))
        
    def forward(self, x):
        return self.model(x)   