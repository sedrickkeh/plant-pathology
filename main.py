import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.nn.functional as F

from sklearn.metrics import f1_score, classification_report, accuracy_score

from preprocess import get_dataloaders
from models import VGGModel, ResNetModel, CustomModel, DenseNetModel
from dataset import PlantDataset


def predict(model, test_loader, device, args, save_file="submission.csv"):
    logits = []
    inds = []

    model.eval()
    for X, ind in test_loader:
        X = X.to(device)
        logit = model(X)
        logits.append(logit.data.cpu().numpy())
        inds.append(ind)
    logits = np.concatenate(logits, axis=0)
    preds = torch.softmax(torch.from_numpy(logits), dim=1).data.cpu()
    inds = np.concatenate(inds, axis=0)
    result = {"image_id":inds, "healthy":preds[:,0], "multiple_diseases":preds[:,1], "rust":preds[:,2], "scab":preds[:,3]}
    df = pd.DataFrame(result)
    df.to_csv(save_file, index=False)  


def trainer(train, val, model, device, optimizer, criterion, args):
    epoch = args.n_epochs
    best_acc = 0
    best_model = model

    for e in range(epoch):
        loss_log = []

        # Training
        model.train()
        pbar = tqdm(enumerate(train),total=len(train))
        for i, (X, y) in pbar:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_log.append(loss.item())
            pbar.set_description("(Epoch {}) TRAIN LOSS:{:.4f}".format((e+1), np.mean(loss_log)))

        # Validation
        model.eval()
        pbar = tqdm(enumerate(val),total=len(val))
        logits = []
        ys = []
        loss_log_dev = []
        for i, (X, y) in pbar:
            X = X.to(device)
            y = y.to(device)
            logit = model(X)
            loss = criterion(logit, y)
            loss_log_dev.append(loss.item())
            logits.append(logit.data.cpu().numpy())
            ys.append(y.data.cpu().numpy())
            pbar.set_description("(Epoch {}) DEV LOSS:{:.4f}".format((e+1), np.mean(loss_log_dev)))
        logits = np.concatenate(logits, axis=0)
        ys = np.concatenate(ys, axis=0)
        ys_onehot = torch.nn.functional.one_hot(torch.as_tensor(ys)) 
        preds = torch.softmax(torch.from_numpy(logits), dim=1).data.cpu()

        acc = roc_auc_score(ys_onehot, preds)

        if acc>best_acc:
            best_acc=acc
            best_model=model

        # else:
        #     early_stop-=1
        print("Fold: {}, Epoch: {}, Current ROC AUC:{}, Best ROC AUC:{}".format(fold, e+1,acc,best_acc))

        
    return best_model, best_acc



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--early_stop", type=int, default=3)
    parser.add_argument("--embed_dim", type=int, default=100)
    parser.add_argument("--class_num", type=int, default=4)
    parser.add_argument("--lr_decay", type=float, default=0.5)
    parser.add_argument("--model", type=str, default="resnet")
    parser.add_argument("--optimizer", type=str, default="adam")
    args = parser.parse_args()

    #setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #load data
    train, val, test = get_dataloaders(args.batch_size)

    #build model
    if (args.model == "resnet"):
        model = ResNetModel(args)
    elif (args.model == "vgg"):
        model = VGGModel(args)
    elif (args.model == "densenet"):
        model = DenseNetModel(args)
    else:
        model = CustomModel(args)
    model.to(device)

    #loss function
    criterion = nn.CrossEntropyLoss()
    #choose optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)
    model, best_acc = trainer(train, val, model, device, optimizer, criterion, args)
    print("Training complete. Preparing to predict on test data")

    predict(model, test, device, args)
    print("Prediction complete. See submission.csv")

    

if __name__== "__main__":
    main()