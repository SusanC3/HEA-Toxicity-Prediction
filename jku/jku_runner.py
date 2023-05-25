import jku_Data
import neural_network

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler

from sklearn.model_selection import KFold

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
max_epochs = 75
LEARNING_RATE = 0.001
dim_input = 801
dim_output = 1

#wandb stuff
wandb.login()
wandb.init(
    project="HEA-Toxicity-Prediction",
    name=f"jku-data",
    config={
        "batch_size": params["batch_size"],
        "epochs": max_epochs,
        "learning_rate": LEARNING_RATE,
        "architecture": "NN",
        "datset": "Full"
    }
)

print("loading data")
dataset = jku_Data.Dataset() 

# pdb.set_trace()

def do_epoch(model, device, X, y, training, optimizer=None):
    if training:
        model.train()
    else:
        model.eval()

    X = X.to(device)
    y = y.to(device)

    if training:
        optimizer.zero_grad()

    pred_result = model(input)

    loss = 0 #calculate roc auc

    if training:
            loss.backward() 
            optimizer.step()

    return loss
     



print("Begin training")

for target in dataset.X_train:

    x_tr = dataset.X_train[target]
    y_tr = dataset.y_train[target]
    x_te = dataset.X_test[target]
    y_te = dataset.y_test[target]

    #make new model for each fold
    model = neural_network.ToxicityRegressor(dim_input, dim_output) #dim input, dim output
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.4, patience=5)

    for epoch in range(max_epochs):
        train_loss = do_epoch(model, device, x_tr, y_tr, True, optimizer=optimizer)
        test_loss = do_epoch(model, device, x_te, y_te, False)
      
        scheduler.step(test_loss)

        wandb.log({"train loss": train_loss, 
                   "test loss": test_loss, 
                   "lr" : optimizer.param_groups[0]['lr'],
                   "epoch": epoch+1})


wandb.finish()


