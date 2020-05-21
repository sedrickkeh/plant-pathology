import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_loss(train_loss, valid_loss, model_name):
    plt.figure()
    plt.ylim(0,1.5)
    sns.lineplot(list(range(len(train_loss))), train_loss)
    sns.lineplot(list(range(len(valid_loss))), valid_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train','Val'])
    plt.savefig('./results/loss_{}.png'.format(model_name))

def plot_acc(train_acc, valid_acc, model_name):
    plt.figure()
    sns.lineplot(list(range(len(train_acc))), train_acc)
    sns.lineplot(list(range(len(valid_acc))), valid_acc)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train','Val'])
    plt.savefig('./results/acc_{}.png'.format(model_name))

def plot_conf_mat(conf_mat, model_name):
    plt.figure()
    labels = ['Healthy', 'Multiple','Rust','Scab']
    sns.heatmap(conf_mat, xticklabels=labels, yticklabels=labels, annot=True)
    plt.savefig('./results/conf_mat_{}.png'.format(model_name))


def save_results(train_loss, val_loss, train_acc, val_acc, model_name):
    train_loss_df = pd.DataFrame(train_loss)
    train_loss_df.to_csv("results/train_loss_{}.csv".format(model_name))
    val_loss_df = pd.DataFrame(val_loss)
    val_loss_df.to_csv("results/val_loss_{}.csv".format(model_name))
    train_acc_df = pd.DataFrame(train_acc)
    train_acc_df.to_csv("results/train_acc_{}.csv".format(model_name))
    val_acc_df = pd.DataFrame(val_acc)
    val_acc_df.to_csv("results/val_acc_{}.csv".format(model_name))