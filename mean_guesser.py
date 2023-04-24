#guesses the mean output value of each feature
#similar to runner just without any actual neural net

import Data
import numpy as np

dataset = Data.Dataset()

#our prediction is just the average of all the outputs
pred = []
for feature in dataset.y.T:
    pred.append(np.mean(feature))

#now calculate the MSE of this prediction with all the outputs
tot_squared_error = 0
for sample in dataset.y:
    tot_squared_error += np.sum(((pred - sample)**2)).item()

print("MSE:", tot_squared_error / (dataset.y.shape[0]*dataset.y.shape[1]))

