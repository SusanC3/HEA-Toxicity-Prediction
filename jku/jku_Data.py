# import input_output
#https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
import numpy as np
import torch
import pickle
import pandas as pd
import pdb

class Dataset:

    #so ik the websit told me to use a different format but I'm not motivated enough to change my current format yet
    def __init__(self):
        y_tr = pd.read_csv('tox21_labels_train.csv', index_col=0)
        y_te = pd.read_csv('tox21_labels_test.csv', index_col=0)
        x_tr_dense = pd.read_csv('tox21_dense_train.csv', index_col=0).values
        x_te_dense = pd.read_csv('tox21_dense_test.csv', index_col=0).values

        self.X_train = {}
        self.y_train = {}
        self.X_test = {}
        self.y_test = {}

        #they do each individually, guess I will too
        for target in y_tr.columns:
            rows_tr = np.isfinite(y_tr[target]).values
            rows_te = np.isfinite(y_te[target]).values

            self.X_train[target] = np.array(x_tr_dense[rows_tr])
            self.y_train[target] = np.array(y_tr[target][rows_tr])
            self.X_test[target] = np.array(x_te_dense[rows_te])
            self.y_test[target] = np.array(y_te[target][rows_te])


    # 'Gets sample at ID index. just index in input/output array'
    # def __getitem__(self, index):
    #     return self.X[index], self.y[index]
