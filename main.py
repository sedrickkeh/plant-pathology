import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import get_cosine_schedule_with_warmup
from transformers import AdamW
from scipy.special import softmax
from tqdm import tqdm

import utils
from models import EfficientNetModel
from dataloader import get_data, get_dataloaders
from trainer import train_fn, valid_fn, test_fn
from plots import plot_loss, plot_acc, plot_conf_mat

def get_model(model_name, args):
    if (model_name == "effnet"):
        return EfficientNetModel(args)
    else return EfficientNetModel(args)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--decay", type=float, default=1e-3)
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--model", type=str, default="effnet")
    args = parser.parse_args()

    ## Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ## Load data
    train_paths, val_paths, test_paths, train_labels, val_labels = get_data()
    train_size = train_labels.shape[0]
    val_size = val_labels.shape[0]
    trainloader, valloader, testloader = get_dataloaders(train_paths, val_paths, test_paths, 
                                                            train_labels, val_labels, args.batch_size)

    ## Initialize Model
    model = get_model(args.model, args)
    model.to(device)

    ## Initialize Optimizer, Scheduler, Loss
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.decay)
    num_train_steps = int(train_size / args.batch_size * args.n_epochs)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=train_size/args.batch_size*5, num_training_steps=num_train_steps)
    loss_func = torch.nn.CrossEntropyLoss()

    ## Training
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    for epoch in range(args.n_epochs): 
        loss_t, acc_t = train_fn(model, trainloader, device, loss_func, optimizer, scheduler, train_size)
        loss_v, acc_v, conf_mat = valid_fn(model, valloader, device, loss_func, optimizer, scheduler, val_size)
        train_loss.append(loss_t)
        val_loss.append(loss_v)
        train_acc.append(acc_t)
        val_acc.append(acc_v)
        
        if (epoch+1) % 10 == 0:
            path = 'epoch' + str(epoch) + '.pt'
            torch.save(model.state_dict(), path)
        
        printstr = 'Epoch: '+ str(epoch) + ', Train loss: ' + str(loss_t) + ', Val loss: ' + str(loss_v) + ', Train acc: ' + str(acc_t) + ', Val acc: ' + str(acc_v)
        tqdm.write(printstr)


    ## Generate plots
    plot_loss(train_loss, val_loss, "effnet")
    plot_acc(train_acc, val_acc, "effnet")
    plot_conf_mat(conf_mat, "effnet")


    ## Testing
    subs = []
    test = pd.read_csv('test.csv')
    for i in range(3): # average over 3 runs
        out = test_fn(model, testloader, device)
        output = pd.DataFrame(softmax(out,1), columns = ['healthy','multiple_diseases','rust','scab'])
        output.drop(0, inplace = True)
        output.reset_index(drop=True, inplace=True)
        subs.append(output)

    sub_eff = sum(subs)/3
    sub1 = sub_eff.copy()
    sub1['image_id'] = test.image_id
    sub1 = sub1[['image_id','healthy','multiple_diseases','rust','scab']]
    sub1.to_csv('submission_efficientnet.csv', index = False)



if __name__== "__main__":
    main()