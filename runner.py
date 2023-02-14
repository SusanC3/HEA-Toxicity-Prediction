import IOWrapper
import neural_network

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from torch.nn import functional as F
import torchvision
from torchvision import datasets,transforms
import torchvision.transforms as transforms

from sklearn.preprocessing import MinMaxScaler    
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

import pdb



#tell pytorch to use GPU if available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

params = {'batch_size': 32,
            'shuffle': True, #shuffle order of data each train
            'num_workers': 6}
max_epochs = 150
LEARNING_RATE = 0.002
dim_input = 801
dim_output = 229432
len_data = 20

print("loading data")
dataset = IOWrapper.Dataset() #__init__ not called for some reason
#training_generator = DataLoader(training_data, **params)

model = neural_network.ToxicityRegressor(dim_input, dim_output) #dim input, dim output
model.to(device)

print(model)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}


#kfold stuff
k = 10
splits = KFold(n_splits=k, shuffle=True, random_state=42)
foldperf={}

def train_epoch(model, device, dataloader, loss_fn, optimizer):
    train_loss, train_correct = 0.0, 0
    model.train()
    
    for input, output in dataloader:
        input, output = input.to(device), output.to(device)
        optimizer.zero_grad()

        pred_result = model(input)
        loss = loss_fn(pred_result, output)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * input.size(0)
        #scores, predictions = torch.max(pred_result.data, 1)
       # pdb.set_trace()
        train_correct += (pred_result == output).sum().item()

    return train_loss, train_correct


def valid_epoch(model, device, dataloader, loss_fn):
    valid_loss, val_correct = 0.0, 0
    model.eval()
    for input, output in dataloader:
        input, output = input.to(device), output.to(device)
        pred_result = model(input)
        loss = loss_fn(pred_result, output)
        valid_loss += loss.item() * input.size(0)
        #scores, predictions = torch.max(pred_result.data, 1)
        val_correct += (pred_result == output).sum().item()

    return valid_loss, val_correct


print("Begin training")

f = open("performance2.txt", "a")

for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len_data))):

    f.write("\n")
    f.write('Fold {}'.format(fold + 1), "\n")

    print("fold", fold + 1)

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(dataset, batch_size=params['batch_size'], sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=params['batch_size'], sampler=test_sampler)

    for epoch in range(max_epochs):
        #print("epoch", epoch + 1)
        train_loss, train_correct = train_epoch(model, device, train_loader, criterion, optimizer)
        test_loss, test_correct = valid_epoch(model, device, test_loader, criterion)

        train_loss /= len(train_loader.sampler)
        train_acc = train_correct / len(train_loader.sampler) * 100
        test_loss /= len(test_loader.sampler)
        test_acc = train_correct / len(test_loader.sampler) * 100

        # print(train_acc)
        # print(test_acc)

        # if (train_acc > 0 or test_acc > 0):
        #     pdb.set_trace()
        


        f.write("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} %\n".format(epoch + 1,
                                                                                                             max_epochs,
                                                                                                             train_loss,
                                                                                                             test_loss,
                                                                                                             train_acc,
                                                                                                             test_acc))

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)


