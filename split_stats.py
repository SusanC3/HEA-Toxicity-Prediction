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


for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(5070))):
    X_train, y_train = [], []
    for id in train_idx:
        X_train.append(dataset.__getitem__(id)[0])
        y_train.append(dataset.__getitem__(id)[1])

    X_test, y_test = [], []
    for id in val_idx:
        X_test.append(dataset.__getitem__(id)[0])
        y_test.append(dataset.__getitem__(id)[1]) 

    print("Fold", str(fold + 1), "output train mean:", str(np.mean(y_train)))
    print("Fold", str(fold + 1), "output test mean:", str(np.mean(y_test)))
    print("Fold", str(fold + 1), "output train variance:", str(np.var(y_train)))
    print("Fold", str(fold + 1), "output test variance:", str(np.var(y_test)))
    print()

    #want variance among each feature --> axis = 0
    plt.title("Variances of Input Features for Fold " + str(fold + 1))
    plt.hist(np.var(X_test, axis=0), label="Test", alpha = 0.7, bins=100)
    plt.hist(np.var(X_train, axis=0), label="Train", alpha = 0.7, bins=100)
    plt.legend()
    plt.savefig("stats_plots/var_"+str(fold+1))
    plt.clf()

    plt.title("Means of Input Features for Fold " + str(fold + 1))
    plt.hist(np.mean(X_test, axis=0), label="Test", alpha = 0.7, bins=100)
    plt.hist(np.mean(X_train, axis=0), label="Train", alpha = 0.7, bins=100)
    plt.legend()
    plt.savefig("stats_plots/mean_"+str(fold+1))
    plt.clf()