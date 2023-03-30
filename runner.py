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
dim_output = 100
len_data = 5070
max_grad_norm = 1

#wandb stuff
wandb.login()
wandb.init(
    project="HEA-Toxicity-Prediction",
    name=f"reduce-lr-on-plateau-test2",
    config={
        "batch_size": params["batch_size"],
        "epochs": max_epochs,
        "learning_rate": LEARNING_RATE,
        "architecture": "NN",
        "datset": "Full"
    }
)

print("loading data")
dataset = Data.Dataset() #__init__ not called for some reason

#kfold stuff
#https://medium.com/dataseries/k-fold-cross-validation-with-pytorch-and-sklearn-d094aa00105f
k = 10
splits = KFold(n_splits=k, shuffle=True, random_state=100)

def do_epoch(model, device, dataloader, training, optimizer=None):
    if training:
        model.train()
    else:
        model.eval()

    tot_squared_error = 0
    counter = 0

    for input, output in dataloader:
        input, output = input.to(device), output.to(device)           
        if training:
            optimizer.zero_grad()

        pred_result = model(input)
        
        #calculate total squared error for this batch, add it to total squared error
        local_se = (pred_result - output)**2
        tot_squared_error += torch.sum(local_se).item()
        counter += local_se.shape[0]*local_se.shape[1]

        if training:
            loss = torch.mean(local_se)
            loss.backward() 

            optimizer.step()


    loss = tot_squared_error / counter #the mean of all the squared errors
    return loss


print("Begin training")

for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len_data))):

    print("fold:", str(fold + 1))

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(dataset, batch_size=params['batch_size'], sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=params['batch_size'], sampler=test_sampler)

    #make new model for each fold
    model = neural_network.ToxicityRegressor(dim_input, dim_output) #dim input, dim output
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    for epoch in range(max_epochs):
        train_loss = do_epoch(model, device, train_loader, True, optimizer=optimizer)
        test_loss = do_epoch(model, device, test_loader, False)

        scheduler.step(test_loss)

        wandb.log({"fold " + str(fold + 1) + " train loss": train_loss, 
                   "fold " + str(fold + 1) + " test loss": test_loss, 
                   "epoch": epoch+1})
        
        #print("epoch", epoch + 1)
        #print("train loss:", train_loss, "test loss:", test_loss)

wandb.finish()


