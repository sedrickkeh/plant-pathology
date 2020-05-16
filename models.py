import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class CustomModel(nn.Module):
    def __init__(self, args):
        super(CustomModel, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, out_channels=64, kernel_size=(3,3), stride=(1,1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d((2,2),stride=(2,2))
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, out_channels=64, kernel_size=(3,3), stride=(1,1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d((2,2),stride=(2,2))
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, out_channels=64, kernel_size=(3,3), stride=(1,1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d((2,2),stride=(2,2))
        )

        self.dropout = nn.Dropout(args.dropout, self.training)
        self.linear1 = nn.Linear(150784, 2048)
        self.linear2 = nn.Linear(2048, 256)
        self.linear3 = nn.Linear(256, 4)


    def forward(self, x):
        x1 = self.layer1(x.float())
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        out = x3.reshape(x3.shape[0], -1)
        logit = F.dropout(out, 0.25, self.training)
        logit = F.relu(self.linear1(logit))
        logit = F.dropout(logit, 0.25, self.training)
        logit = F.relu(self.linear2(logit))
        logit = F.dropout(logit, 0.25, self.training)
        logit = self.linear3(logit)

        return logit

class ResNetModel(nn.Module):
    def __init__(self, args):
        super(ResNetModel, self).__init__()
        self.backbone = torchvision.models.resnet101(pretrained=True)
        print(self.backbone)
        in_features = self.backbone.fc.in_features
        self.logit = nn.Linear(in_features, args.class_num)
        
    def forward(self, x):
        batch_size, C, H, W = x.shape
        x = self.backbone.conv1(x.float())
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = F.adaptive_avg_pool2d(x,1).reshape(batch_size,-1)
        x = F.dropout(x, 0.25, self.training)

        x = self.logit(x)
        return x

class VGGModel(nn.Module):
    def __init__(self, args):
        super(VGGModel, self).__init__()
        self.backbone = torchvision.models.vgg16(pretrained=True)
        print(self.backbone)
        self.in_features = self.backbone.classifier[6].out_features
        self.logit = nn.Linear(self.in_features, args.class_num)
        self.logit2 = nn.Linear(512*7*7, args.class_num)

    def forward(self, x):
        batch_size, C, H, W = x.shape
        x = self.backbone.features(x.float())
        x = self.backbone.avgpool(x)
        x = x.reshape(batch_size, -1)
        # x = self.backbone.classifier(x)
        # x = F.dropout(x, 0.25, self.training)
        x = self.logit2(x)
        return x

class DenseNetModel(nn.Module):
    def __init__(self, args):
        super(DenseNetModel, self).__init__()
        self.backbone = torchvision.models.densenet161(pretrained=True)
        print(self.backbone)
        in_features = self.backbone.classifier.in_features
        self.logit = nn.Linear(in_features, args.class_num)
    
    def forward(self, x):
        batch_size, C, H, W = x.shape
        x = self.backbone.features(x.float())
        x = F.adaptive_avg_pool2d(x,1).reshape(batch_size,-1)
        x = self.logit(x)
        return x

