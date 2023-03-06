import Data
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
import math
import wandb



#tell pytorch to use GPU if available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)
torch.backends.cudnn.benchmark = True

params = {'batch_size': 64,
            'shuffle': True, #shuffle order of data each train
            'num_workers': 6}
max_epochs = 150
LEARNING_RATE = 0.002
dim_input = 801
dim_output = 229432
len_data = 5070

#wandb stuff
# wandb.login()
# wandb.init(
#     project="HEA-Toxicity-Prediction",
#     name=f"experiment_strange_loss",
#     config={
#         "batch_size": params["batch_size"],
#         "epochs": max_epochs,
#         "learning_rate": LEARNING_RATE,
#         "architecture": "NN",
#         "datset": "Full"
#     }
# )

print("loading data")
dataset = Data.Dataset() #__init__ not called for some reason

#training_generator = DataLoader(training_data, **params)

model = neural_network.ToxicityRegressor(dim_input, dim_output) #dim input, dim output
model.to(device)

print(model)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}


#kfold stuff
#https://medium.com/dataseries/k-fold-cross-validation-with-pytorch-and-sklearn-d094aa00105f
k = 10
splits = KFold(n_splits=k, shuffle=True, random_state=1)
foldperf={}

def train_epoch(model, device, dataloader, loss_fn, optimizer):
    train_loss, train_correct = 0.0, 0
    model.train()
    
    for input, output in dataloader:
        input, output = input.to(device), output.to(device)     
        optimizer.zero_grad()

        pred_result = model(input)
        loss = loss_fn(pred_result, output)
      #  print("before backward", loss.item())
        loss.backward()
        optimizer.step()
      #  print("after backward", loss.item)
        train_loss += loss.item()
        # train_correct += (pred_result == output).sum().item()

#
    train_loss /= len(dataloader)
    return train_loss, train_correct


def valid_epoch(model, device, dataloader, loss_fn):
    valid_loss, val_correct = 0.0, 0
    model.eval()

    tot_squared_error = 0
    counter = 0

    for input, output in dataloader:
        input, output = input.to(device), output.to(device)           
       
        pred_result = model(input)
        
        #calculate total squared error for this batch, add it to total squared error
        local_se = (pred_result - output)**2
        tot_squared_error += torch.sum(local_se).item()
        counter += local_se.shape[0]*local_se.shape[1]

      #  pdb.set_trace()
        #loss = torch.mean( (pred_result - output)**2) #loss_fn(pred_result, output)
        #valid_loss += loss.item()

    #valid_loss /= len(dataloader)
    valid_loss = tot_squared_error / counter #the mean of all the squared errors
    return valid_loss, val_correct


print("Begin training")

f = open("performance2.txt", "a")

for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len_data))):

    # f.write("\n")
    # f.write('Fold {}'.format(fold + 1), "\n")


    print("fold", fold + 1)

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(val_idx)
    # train_size = int(0.8 * len(dataset))
    # test_size = len(dataset) - train_size  
    # train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(1))  
    
    train_loader = DataLoader(dataset, batch_size=params['batch_size'], sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=params['batch_size'], sampler=test_sampler)

    test_loss, test_correct = valid_epoch(model, device, test_loader, criterion)
    print("valid test loss", test_loss)
    train_loss, train_correct = valid_epoch(model, device, train_loader, criterion)
    print("valid train loss", train_loss)

    pdb.set_trace()


    # print()
    
    # train_test_loss, train_test_correct = train_epoch(model, device, test_loader, criterion, optimizer)
    # print("train test loss", train_test_loss)
    # train_train_loss, train_train_correct = train_epoch(model, device, train_loader, criterion, optimizer)
    # print("train train loss", train_train_loss)

    

    for epoch in range(max_epochs):
        print("epoch", epoch + 1)
        train_loss, train_correct = train_epoch(model, device, train_loader, criterion, optimizer)
        test_loss, test_correct = valid_epoch(model, device, test_loader, criterion)
        
        train_acc = train_correct / len(train_loader.sampler) * 100
        test_acc = train_correct / len(test_loader.sampler) * 100
        

        #don't want to log all hyperparameters, for now i'll just log the max and the average param

        if (math.isinf(test_loss) or math.isnan(test_loss)):
            pdb.set_trace()

        wandb.log({"train_loss": train_loss, "test_loss": test_loss, "train_acc": train_acc, "test_acc": test_acc})
        print("train loss:", train_loss, "test loss:", test_loss)
      
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)



wandb.finish()


