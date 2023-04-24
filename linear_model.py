from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler, DataLoader
import numpy as np
import Data
import pdb
import matplotlib.pyplot as plt

dataset = Data.Dataset()

k = 10
splits = KFold(n_splits=k, shuffle=True, random_state=100)

mse_train = []
mse_test = []

# #will fit to last 90% of data, score on first 10%
X_train = dataset.X[int(0.1*len(dataset.X)):]
y_train = dataset.y[int(0.1*len(dataset.y)):]

X_test = dataset.X[:int(0.1*len(dataset.X))]
y_test = dataset.y[:int(0.1*len(dataset.y))]

model = Ridge()
model.fit(X_train, y_train)

preds_fit = model.predict(X_train)
tot_squared_error = np.sum((preds_fit - y_train)**2)
denominator = preds_fit.shape[0]*preds_fit.shape[1]
print("MSE train:", tot_squared_error/denominator)
mse_train.append(tot_squared_error/denominator)

preds_score = model.predict(X_test)
tot_squared_error = np.sum((preds_score - y_test)**2)
denominator = preds_score.shape[0]*preds_score.shape[1]
print("MSE test:", tot_squared_error/denominator)
mse_test.append(tot_squared_error/denominator)



def MSE_from_alpha(a):
    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(5070))):

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
        denominator = preds_fit.shape[0]*preds_fit.shape[1]
    #  print("MSE train:", tot_squared_error/denominator)
        mse_train.append(tot_squared_error/denominator)

        preds_score = model.predict(X_test)
        tot_squared_error = np.sum((preds_score - y_test)**2)
        denominator = preds_score.shape[0]*preds_score.shape[1]
    # print("MSE test:", tot_squared_error/denominator)
        mse_test.append(tot_squared_error/denominator)
        
    # print()

    print("Average MSE train:", np.mean(np.array(mse_train)))
    print("Average MSE test:", np.mean(np.array(mse_test)))
    print()
    return np.mean(np.array(mse_train)), np.mean(np.array(mse_test))

#for plotting
# alphas = []
# train = []
# test = []

# # for i in range(-5, 60):
# #     alpha = 1 * 10**i
# #     print("doing alpha", alpha)
# #     train_mse, test_mse = MSE_from_alpha(alpha)
# #     alphas.append(alpha)
# #     train.append(train_mse)
# #     test.append(test_mse)

# print("showing plot")
# plt.plot(alphas, train, label = "train MSE")
# plt.plot(alphas, test, label = "test MSE")
# plt.legend()
# plt.title("Train and test MSE vs alpha value (log scale on both axes)")
# plt.xscale("log")
# plt.yscale("log")
# plt.xlabel("Alpha")
# plt.ylabel("MSE")
# plt.savefig("Error vs Alpha")