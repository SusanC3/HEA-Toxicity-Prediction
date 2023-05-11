from sklearn.model_selection import KFold
import numpy as np
import Data
import pdb
import matplotlib.pyplot as plt
import math

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

    print("output mean percent diff:", str(100*abs(np.mean(y_train) - np.mean(y_test)) / abs(np.mean(y_train))))
    print("output variance percent diff:", str(100*abs(np.var(y_train) - np.var(y_test)) / abs(np.var(y_train))))
    print()

    var_train = np.var(X_train, axis = 0)
    var_test = np.var(X_test, axis = 0)
    var_with_nan = 100*abs(var_train - var_test) / abs(var_train)
    var_perc_diff = var_with_nan[~np.isnan(var_with_nan)]
    for i in range(len(var_perc_diff)):
        if var_perc_diff[i] > 100:
            var_perc_diff[i] = 100

    #want variance among each feature --> axis = 0
    plt.title("Percent difference between train and test variances for fold " + str(fold + 1))
    #plt.xscale("log")
    plt.hist(var_perc_diff, bins=100)
    plt.savefig("stats_plots/var_"+str(fold+1))
    plt.clf()

    mean_train = np.mean(X_train, axis = 0)
    mean_test = np.mean(X_test, axis = 0)
    mean_with_nan = abs(mean_train - mean_test) / abs(var_test)
    mean_perc_diff = mean_with_nan[~np.isnan(mean_with_nan)]
    for i in range(len(mean_perc_diff)):
        if mean_perc_diff[i] > 100:
            mean_perc_diff[i] = 100

    plt.title("Percent difference between train and test means for fold  " + str(fold + 1))
    #plt.xscale("log")
    plt.hist(mean_perc_diff, bins=100)
    plt.savefig("stats_plots/mean_"+str(fold+1))
    plt.clf()
    print()

    #pdb.set_trace()
