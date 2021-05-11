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

from Dconv import BesselConv2d, Dconv2d
from qclosing import Closing2d
from qdilation import Dilation2d

from utils import *

device = torch.device('cuda')

batch_size = 200

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# get the dataset directories and divide it into a list of training and testing datsets
# dataset is MNIST Large Scale from http://www.diva-portal.org/smash/record.jsf?pid=diva2%3A1432166&dswid=-7936
dataset_dir =  '/mnist_large_scale'

file_list = glob.glob(dataset_dir+'/*.h5')
file_list = [name[len(dataset_dir):] for name in file_list]
train_list = [name for name in file_list if name[19]=='r']
test_list = [name for name in file_list if name[19]=='e']
test_list.sort()
test_size = 10000



class IdLifting(nn.Module):
    def __init__(self, n_scales):
        super(IdLifting, self).__init__()
        self.n_scales = n_scales
    def forward(self, x):
        return torch.stack([x]*self.n_scales, 2)

class ModelEq(nn.Module):
    def __init__(self, padding=1, reduction='max', lifting='dilation'):
        super(ModelEq, self).__init__()
        n_scales = 6
        base = 2
        zero_scale = 1/4

        
        self.reduction = reduction
        self.padding = padding
        
        # define the lifting layer
        if lifting == 'gaussian':
            self.lifting = BesselConv2d(zero_scale=zero_scale, base=base, n_scales=n_scales)
        if lifting == 'dilation':
            self.lifting = Dilation2d(zero_scale=zero_scale, base=base, n_scales=n_scales)
        if lifting == 'closing':
            self.lifting = Closing2d(zero_scale=zero_scale, base=base, n_scales=n_scales)
        if lifting == 'id':
            self.lifting = IdLifting(n_scales)

        # define the semigroup correlation layers
        self.conv1 = Dconv2d(1, 16, (1,3,3), 2, [n_scales, n_scales], stride=1, padding=padding, bias=True)
        self.conv2 = Dconv2d(16, 16, (1,3,3), 2, [n_scales, n_scales], stride=2, padding=padding, bias=True)
        self.conv3 = Dconv2d(16, 32, (1,3,3), 2, [n_scales, n_scales], stride=1, padding=padding, bias=True)
        self.conv4 = Dconv2d(32, 32, (1,3,3), 2, [n_scales, n_scales], stride=2, padding=padding, bias=True)
        self.conv5 = Dconv2d(32, 64, (1,3,3), 2, [n_scales, n_scales], stride=1, padding=padding, bias=True)

        self.bn1 = nn.BatchNorm3d(16, momentum=0.005)
        self.bn2 = nn.BatchNorm3d(16, momentum=0.005)
        self.bn3 = nn.BatchNorm3d(32, momentum=0.005)
        self.bn4 = nn.BatchNorm3d(32, momentum=0.005)
        self.bn5 = nn.BatchNorm3d(64, momentum=0.005)
        
    def forward(self, x):
        x = self.lifting(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        
        if self.reduction == 'max':
            x = torch.max(x, 2)[0]
            x = torch.max(x, 2)[0]
            x = torch.max(x, 2)[0]
        else:
            x = torch.mean(x, [2,3,4])
            
        x = torch.flatten(x, 1)
        return x



lifting = 'dilation'
#lifting = 'gaussian'
#lifting = 'closing'
#lifting = 'id'
reduction = 'max'

train_name =  'mnist_large_scale_tr50000_vl10000_te10000_outsize112-112_sctr%dp000_scte%dp000.h5'
train_scale = 2

x_train, y_train, x_val, y_val = load_train_and_val(os.path.join(dataset_dir, train_name % (train_scale, train_scale)))
# change data to channels first
x_train = x_train.transpose(0, 3, 1, 2)
x_val = x_val.transpose(0, 3, 1, 2)

# create pytorch datasets
trainset = torch.utils.data.TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
valset = torch.utils.data.TensorDataset(torch.tensor(x_val), torch.tensor(y_val))

train_size = len(trainset)
val_size = len(valset)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True)

model = torch.jit.trace(ModelEq(0, reduction, lifting).to(device), [torch.rand(256, 1, 112, 112).cuda()])
train(model, train_loader, train_size, val_loader, val_size, nn.CrossEntropyLoss(), 1e-2, 50, 5, .95, 1, 1e-4, verbose=1)
#torch.save(model.state_dict(), 'checkpoints_test_invariance/%s_tr%d_%s.ckpt' % (lifting, train_scale, reduction))
acc_list = list()
for filename in test_list:
    x_test, y_test = load_test(os.path.join(dataset_dir, filename))
    x_test = x_test.transpose(0, 3, 1, 2)
    testset = torch.utils.data.TensorDataset(torch.tensor(x_test), torch.tensor(y_test))
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

    acc_test = test(model, test_loader, test_size)
    print(f'test accuracy: {acc_test}')
    acc_list.append(acc_test)
    #np.save('accuracies/accuracy_%s_tr%d_%s' % (lifting, train_scale, reduction), np.array(acc_list))
