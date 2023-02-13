import IOWrapper
import neural_network

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import pdb

#tell pytorch to use GPU if available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

params = {'batch_size': 16,
            'shuffle': True, #shuffle order of data each train
            'num_workers': 6}
max_epochs = 150
LEARNING_RATE = 0.001
NUM_FEATURES = 801

training_data = IOWrapper.Dataset() #__init__ not called for some reason
training_generator = DataLoader(training_data, **params)

#i'll worry about validation later

model = neural_network.ToxicityRegressor(NUM_FEATURES)
model.to(device)

print(model)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

loss_stats = [0 for i in range(max_epochs)]

print("Begin training")
model.train() #we're in training mode
for epoch in range(max_epochs):
    print("on epoch", epoch, "out of", max_epochs)
    for input, output in training_generator:
        #transfer to GPU
        local_input, local_output = input.to(device), output.to(device)
        optimizer.zero_grad()

     #   pdb.set_trace()

        y_train_pred = model(local_input)
        train_loss = criterion(y_train_pred, local_output.unsqueeze(1))
        train_loss.backward()
        optimizer.step()

        



































# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from scikeras.wrappers import KerasRegressor
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline

# import input_output

# X, Y = input_output.get_input_output() #TODO get from feature selection

# print("defining model")
# def get_model():
#     model = Sequential()
#     model.add(Dense(5070, input_shape=(5070,), kernel_initializer='normal', activation='relu'))
#     model.add(Dense(1, kernel_initializer='normal'))
#     model.compile(loss='mean_squared_error', optimizer='adam')
#     return model


# print("standardizing data")
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(model=get_model, epochs=50, batch_size = 128, verbose=1)))
# pipeline = Pipeline(estimators)
# print("Starting training & assessment")
# #use kfold for now, in the futre want to use whole data with just this test data as test
# kfold = KFold(n_splits=10)
# print("After kfold thing")
# results = cross_val_score(pipeline, X, Y, cv=kfold, scoring='neg_mean_squared_error')
# print('Results:', results.mean(), "(", results.std(), ") MSE")
# #TODO WILL NEED TO RE-SCALE OUTPUT FOR ACTUALLY CORRECT GUESSES