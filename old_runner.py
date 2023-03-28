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
LEARNING_RATE = 0.001
dim_input = 801
dim_output = 229432
len_data = 5070
max_grad_norm = 1

#wandb stuff
# wandb.login()
# wandb.init(
#     project="HEA-Toxicity-Prediction",
#     name=f"clip_grad_norm",
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

#kfold stuff
#https://medium.com/dataseries/k-fold-cross-validation-with-pytorch-and-sklearn-d094aa00105f
k = 10
splits = KFold(n_splits=k, shuffle=True, random_state=27)
foldperf={}

def train_epoch(model, device, dataloader, loss_fn, optimizer):
    train_loss = 0.0
    pre_clip_grad_norms = np.zeros(model.n_layers)
    post_clip_grad_norms = np.zeros(model.n_layers)
    model.train()

    tot_squared_error = 0
    counter = 0
    
    for input, output in dataloader:
        input, output = input.to(device), output.to(device)     
        optimizer.zero_grad()

        pred_result = model(input)

        #calculate total squared error for this batch, add it to total squared error
        local_se = (pred_result - output)**2
        tot_squared_error += torch.sum(local_se).item()
        counter += local_se.shape[0]*local_se.shape[1]

        loss = loss_fn(pred_result, output)
        loss.backward()

        #clip gradient norm
        pre_clip_grad_norms += model.compute_grad_norm()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        post_clip_grad_norms += model.compute_grad_norm()

        optimizer.step()
       # train_loss += loss.item()

    train_loss = tot_squared_error / counter
    for i in range(len(pre_clip_grad_norms)):    
        wandb.log({"pre clip layer " + str(i+1): pre_clip_grad_norms[i]})
        wandb.log({"post clip layer " + str(i+1): post_clip_grad_norms[i]})
    return train_loss


def valid_epoch(model, device, dataloader, loss_fn):
    valid_loss = 0.0
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

        #loss = torch.mean( (pred_result - output)**2) #loss_fn(pred_result, output)
        #valid_loss += loss.item()

    #valid_loss /= len(dataloader)
    valid_loss = tot_squared_error / counter #the mean of all the squared errors
    return valid_loss


print("Begin training")

f = open("performance2.txt", "a")

for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len_data))):

    print("fold", fold + 1)

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(dataset, batch_size=params['batch_size'], sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=params['batch_size'], sampler=test_sampler)

    #make new model for each fold
    model = neural_network.ToxicityRegressor(dim_input, dim_output) #dim input, dim output
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    test_loss = valid_epoch(model, device, test_loader, criterion)
    train_loss = valid_epoch(model, device, train_loader, criterion)

    print("test loss", test_loss)
    print("train loss", train_loss)

    pdb.set_trace()

    for epoch in range(max_epochs):
        print("epoch", epoch + 1)
        train_loss = train_epoch(model, device, train_loader, criterion, optimizer)
        test_loss = valid_epoch(model, device, test_loader, criterion)

        wandb.log({"fold " + str(fold + 1) + " train loss": train_loss, 
                   "fold " + str(fold + 1) + " test loss": test_loss, 
                   "epoch": epoch+1})
        print("train loss:", train_loss, "test loss:", test_loss)


wandb.finish()

