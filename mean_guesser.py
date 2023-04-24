#guesses the mean output value of each feature
#similar to runner just without any actual neural net

import Data
import numpy as np

dataset = Data.Dataset()

X_train = dataset.X[:int(0.9*len(dataset.X))]
y_train = dataset.y[:int(0.9*len(dataset.y))]

X_test = dataset.X[int(0.9*len(dataset.X)):]
y_test = dataset.y[int(0.9*len(dataset.y)):]

#our prediction is just the average of all the outputs
pred_train = []
for feature in y_train.T:
    pred_train.append(np.mean(feature))

pred_test = []
for feature in y_test.T:
    pred_test.append(np.mean(feature))

#now calculate the MSE of this prediction with all the outputs
tot_squared_error = 0
for sample in y_train:
    tot_squared_error += np.sum(((pred_train - sample)**2)).item()
MSE_train = tot_squared_error / (y_train.shape[0]*y_train.shape[1])

tot_squared_error = 0
for sample in y_test:
    tot_squared_error += np.sum(((pred_test - sample)**2)).item()
MSE_test = tot_squared_error / (y_test.shape[0]*y_test.shape[1])

print("MSE train:", MSE_train)
print("MSE test:", MSE_test)