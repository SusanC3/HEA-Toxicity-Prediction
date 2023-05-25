from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import numpy as np
import Data
import pdb
import matplotlib.pyplot as plt

print("loading data")
dataset = Data.Dataset()

k = 2
splits = KFold(n_splits=k, shuffle=True, random_state=100)


def MSE_from_alpha(a):
    mse_tests = []
    mse_trains = []
   
    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(1724))):
        X_train, y_train = [], []
        for id in train_idx:
            X_train.append(dataset.__getitem__(id)[0])
            y_train.append(dataset.__getitem__(id)[1])

        X_test, y_test = [], []
        for id in val_idx:
            X_test.append(dataset.__getitem__(id)[0])
            y_test.append(dataset.__getitem__(id)[1]) 

        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        #only use input normalizer for normalization

        # print("X train mean:", np.mean(X_train))
        # print("X train var:", np.var(X_train))
        # print("X test mean:", np.mean(X_test))
        # print("X test var:", np.var(X_test))
        # print()
        # print("y train mean:", np.mean(y_train))
        # print("y train var:", np.var(y_train))
        # print("y test mean:", np.mean(y_test))
        # print("y test var:", np.var(y_test))
        # print()

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
        # if (tot_squared_error/denominator < 1):
        mse_tests.append(tot_squared_error/denominator)
       # pdb.set_trace()

        print()

        plt.hist(model.predict(X_test), bins=100)
        plt.savefig("predictions")
        plt.clf()

        plt.hist(y_test, bins=100)
        plt.savefig("actual")
        plt.clf()

        pdb.set_trace()

        #our prediction is just the average of all the outputs
        # pred_train = []
        # for feature in y_train.T:
        #     pred_train.append(np.mean(feature))

        # pred_test = []
        # for feature in y_test.T:
        #     pred_test.append(np.mean(feature))

        # #now calculate the MSE of this prediction with all the outputs
        # tot_squared_error = 0
        # for sample in y_train:
        #     tot_squared_error += np.sum(((pred_train - sample)**2)).item()
        # MSE_train = tot_squared_error / (y_train.shape[0])

        # tot_squared_error = 0
        # for sample in y_test:
        #     tot_squared_error += np.sum(((pred_test - sample)**2)).item()
        # MSE_test = tot_squared_error / (y_test.shape[0])

        # print("MSE train:", MSE_train)
        # print("MSE test:", MSE_test)

        # pdb.set_trace()
  #  pdb.set_trace()
    return np.mean(np.array(mse_trains)), np.mean(np.array(mse_tests))
      

print("Starting fit")
MSE_from_alpha(500)

# print("Mean MSE train:", np.mean(np.array(mse_trains)))
# print("Mean MSE test:", np.mean(np.array(mse_tests)))

# alphas = []
# trains = []
# tests = []

# for i in range(6):
#     print("alpha = ", 10**i)
#     train, test = MSE_from_alpha(10**i)
#     alphas.append(10**i)
#     trains.append(train)
#     tests.append(test)

# pdb.set_trace()

# print("plotting")
# plt.plot(alphas, trains, label="train loss")
# plt.plot(alphas, tests, label="tests loss")
# plt.xscale("log")
# plt.savefig("plot")
