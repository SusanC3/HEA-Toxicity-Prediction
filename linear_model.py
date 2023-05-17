from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import numpy as np
import Data
import pdb
import matplotlib.pyplot as plt

print("loading data")
dataset = Data.Dataset()

k = 10
splits = KFold(n_splits=k, shuffle=True, random_state=100)

mse_tests = []
mse_trains = []

def MSE_from_alpha(a):
    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(1724))):
        X_train, y_train = [], []
        for id in train_idx:
            X_train.append(dataset.__getitem__(id)[0])
            y_train.append(dataset.__getitem__(id)[1])

        X_test, y_test = [], []
        for id in val_idx:
            X_test.append(dataset.__getitem__(id)[0])
            y_test.append(dataset.__getitem__(id)[1]) 

        model = Ridge(alpha=a)
        model.fit(X_train, y_train)

        preds_fit = model.predict(X_train)
        tot_squared_error = np.sum((preds_fit - y_train)**2)
        denominator = preds_fit.shape[0]#*preds_fit.shape[1]
        print("MSE train:", tot_squared_error/denominator)
        mse_trains.append(tot_squared_error/denominator)

        preds_score = model.predict(X_test)
        tot_squared_error = np.sum((preds_score - y_test)**2)
        denominator = preds_score.shape[0]#*preds_score.shape[1]
        print("MSE test:", tot_squared_error/denominator)
        if (tot_squared_error/denominator < 1):
          mse_tests.append(tot_squared_error/denominator)

        print()
      

print("Starting fit")
MSE_from_alpha(1)

print("Mean MSE train:", np.mean(np.array(mse_trains)))
print("Mean MSE test:", np.mean(np.array(mse_tests)))