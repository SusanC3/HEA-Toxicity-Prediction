import jku_Data
import neural_network
import Normalizer

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

import pdb
import math
import wandb
import matplotlib.pyplot as plt


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
model = "Linear Regressor"

#wandb stuff
# wandb.login()
# wandb.init(
#     project="HEA-Toxicity-Prediction",
#     name=f"jku-data",
#     config={
#         "batch_size": params["batch_size"],
#         "epochs": max_epochs,
#         "learning_rate": LEARNING_RATE,
#         "architecture": "NN",
#         "datset": "Full"
#     }
# )

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

    pred_result = model(X.float()).squeeze()

    loss_calc = nn.BCELoss()
    loss = loss_calc(pred_result, y.float())

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

    if model == neural_network:
        #make new model for each fold
        model = neural_network.ToxicityRegressor(dim_input, dim_output) #dim input, dim output
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.4, patience=5)

        for epoch in range(max_epochs):
            

            train_loss = do_epoch(model, device, torch.from_numpy(x_tr), torch.from_numpy(y_tr), True, optimizer=optimizer)
            test_loss = do_epoch(model, device, torch.from_numpy(x_te), torch.from_numpy(y_te), False)
        
            scheduler.step(test_loss)

            wandb.log({"train BCE": train_loss, 
                       "test BCE": test_loss, 
                       "lr" : optimizer.param_groups[0]['lr'],
                       "epoch": epoch+1})

        y_pred = model(torch.from_numpy(x_te).to(device).float())  
        roc_auc = roc_auc_score(y_te, y_pred.cpu().detach())
        print(target, ":", roc_auc_score(y_te, test_pred))
    elif model == "Linear Regressor":
        normalizer = Normalizer.UnitGaussianNormalizer(torch.from_numpy(x_tr))
        x_tr_norm = normalizer.encode(torch.from_numpy(x_tr))
        x_te_norm = normalizer.encode(torch.from_numpy(x_te))
        
        clf = LogisticRegression(random_state=0, max_iter=1000).fit(x_tr_norm, y_tr)
        train_pred = clf.predict(x_tr_norm)
        test_pred = clf.predict(x_te_norm)
        # print("train score:", clf.score(x_tr_norm, y_tr))
        # print("test score:", clf.score(x_te_norm, y_te))

        y_pred_proba = clf.predict_proba(x_te_norm)[::,1]
        fpr, tpr, _ = metrics.roc_curve(y_te, y_pred_proba)

        plt.plot(fpr, tpr)
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.title(target)
        plt.savefig(str(target) + ".png")
        plt.clf()

  


wandb.finish()

