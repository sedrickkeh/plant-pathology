import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix


def train_fn(model, loader, device, loss_func, optimizer, scheduler, train_size):  
    running_loss = 0
    preds_for_acc = []
    labels_for_acc = []
    
    pbar = tqdm(total = len(loader), desc='Training')
    
    for _, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        model.train()
        optimizer.zero_grad()
        predictions = model(images)
        loss = loss_func(predictions, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item()*labels.shape[0]
        labels_for_acc = np.concatenate((labels_for_acc, labels.cpu().numpy()), 0)
        preds_for_acc = np.concatenate((preds_for_acc, np.argmax(predictions.cpu().detach().numpy(), 1)), 0)
        
        pbar.update()
        
    accuracy = accuracy_score(labels_for_acc, preds_for_acc)
    pbar.close()
    return running_loss/train_size, accuracy


def valid_fn(model, loader, device, loss_func, optimizer, scheduler, val_size): 
    running_loss = 0
    preds_for_acc = []
    labels_for_acc = []
    
    pbar = tqdm(total = len(loader), desc='Validation')
    
    with torch.no_grad():       
        for _, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            model.eval()
            predictions = model(images)
            loss = loss_func(predictions, labels)

            running_loss += loss.item()*labels.shape[0]
            labels_for_acc = np.concatenate((labels_for_acc, labels.cpu().numpy()), 0)
            preds_for_acc = np.concatenate((preds_for_acc, np.argmax(predictions.cpu().detach().numpy(), 1)), 0)
            
            pbar.update()
            
        accuracy = accuracy_score(labels_for_acc, preds_for_acc)
        conf_mat = confusion_matrix(labels_for_acc, preds_for_acc)
    
    pbar.close()
    return running_loss/val_size, accuracy, conf_mat


def test_fn(model, loader, device):
    preds_for_output = np.zeros((1,4))
    
    with torch.no_grad():
        pbar = tqdm(total = len(loader))
        for _, images in enumerate(loader):
            images = images.to(device)
            model.eval()
            predictions = model(images)
            preds_for_output = np.concatenate((preds_for_output, predictions.cpu().detach().numpy()), 0)
            pbar.update()
    
    pbar.close()
    return preds_for_output
