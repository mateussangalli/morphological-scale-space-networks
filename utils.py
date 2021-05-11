import numpy as np
import h5py
import glob
import os
import matplotlib.pyplot as plt
import copy
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

def train(model, train_loader, train_size, test_loader, test_size, loss_fn, learning_rate, maxiter, patience, lr_decay, lr_decay_rate, min_lr, verbose=1):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # accuracies history
    acc_train_list = list()
    acc_val_list = list()

    early_stop_counter = 0

    for t in range(maxiter):
        model.train()
        time0 = time.time()
        loss_sum = 0
        acc_sum = 0
        for x, y in train_loader:
            y_pred = model(x.cuda())

            loss = loss_fn(y_pred, y.cuda())
            loss_sum += loss.item()

            y_pred_classes = y_pred.argmax(1)
            acc_sum += (y_pred_classes==y.cuda()).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_mean = loss_sum/math.ceil(train_size/x.size()[0])
        acc_mean = acc_sum.float()/train_size
        acc_train_list.append(acc_mean)


        if lr_decay_rate and t>0 and t%lr_decay_rate==0 and learning_rate>min_lr:
            learning_rate = max(learning_rate*lr_decay, min_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

        model.eval()
        acc_sum = 0
        for x, y in test_loader:
            y_pred = model(x.cuda())
            y_pred = y_pred.argmax(1)
            y = y.cuda()
            acc_sum += (y_pred==y).sum()

        acc_val = acc_sum.float()/test_size
        if len(acc_val_list) > 0 and acc_val < acc_val_list[-1]:
            early_stop_counter += 1
        else:
            early_stop_counter = 0    
        acc_val_list.append(acc_val)
        if early_stop_counter > patience:
            print('early stopping...')
            print('epoch: %d, time: %.2fs, loss: %.4f, acc: %.4f, val acc: %.4f' % (t, delta_time, loss_mean, acc_mean, acc_val))
            break

        delta_time = time.time() - time0
        if verbose and (t%verbose==0 or t==maxiter-1):
            print('epoch: %d, time: %.2fs, loss: %.4f, acc: %.4f, val acc: %.4f' % (t, delta_time, loss_mean, acc_mean, acc_val))

    return acc_train_list, acc_val_list


def test(model, test_loader, test_size):
    model.eval()
    acc_sum = 0
    for x, y in test_loader:
        y_pred = model(x.cuda())
        y_pred = y_pred.argmax(1)
        y = y.cuda()
        acc_sum += (y_pred==y).sum()
    acc_val = acc_sum.float()/test_size
    return acc_val.item()

class TensorDatasetTransform(torch.utils.data.Dataset):
    def __init__(self, tensor1, tensor2, transform=None):
        self.tensor1 = tensor1
        self.tensor2 = tensor2
        self.transform = transform

    def __getitem__(self, index):
        if self.transform is not None:
            return (self.transform(self.tensor1[index]), self.tensor2[index])
        else:
            return (self.tensor1[index], self.tensor2[index])

    def __len__(self):
        return self.tensor1.size(0)

def load_train_and_val(filename):
    print('loading training and validation sets from ' + filename)
    with h5py.File(filename, 'r') as f:
        x_train = np.array( f["/x_train"], dtype=np.float32)
        x_val = np.array( f["/x_val"], dtype=np.float32)

        y_train = np.array( f["/y_train"], dtype=np.int64).reshape(-1)
        y_val = np.array( f["/y_val"], dtype=np.int64).reshape(-1)
    return x_train, y_train, x_val, y_val

def load_test(filename):
    print('loading test set from ' + filename)
    with h5py.File(filename, 'r') as f:
        x_test = np.array( f["/x_test"], dtype=np.float32)
        y_test = np.array( f["/y_test"], dtype=np.int64).reshape(-1)
    return x_test, y_test

