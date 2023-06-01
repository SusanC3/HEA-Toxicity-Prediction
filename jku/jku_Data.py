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

        self.rf_bces_train = {}
        self.rf_bces_test = {}
        self.lr_bces_train = {}
        self.lr_bces_test = {}

        #they do each individually, guess I will too
        for target in y_tr.columns:
            rows_tr = np.isfinite(y_tr[target]).values
            rows_te = np.isfinite(y_te[target]).values

            self.X_train[target] = x_tr_dense[rows_tr]
            self.y_train[target] = y_tr[target][rows_tr]
            self.X_test[target] = x_te_dense[rows_te]
            self.y_test[target] = y_te[target][rows_te]


        #linear regression on training data
        self.lr_bces_train["NR.AhR"] = 6.5039687240848245
        self.lr_bces_train["NR.AR"] = 1.8645121193287757
        self.lr_bces_train["NR.AR.LBD"] = 1.174875734297334
        self.lr_bces_train["NR.Aromatase"] = 3.172043010752688
        self.lr_bces_train["NR.ER"] = 8.314493845767394
        self.lr_bces_train["NR.ER.LBD"] = 2.7101769911504423
        self.lr_bces_train["NR.PPAR.gamma"] = 1.739439119630813
        self.lr_bces_train["SR.ARE"] = 10.174300770166194
        self.lr_bces_train["SR.ATAD5"] = 2.7021253871622344
        self.lr_bces_train["SR.HSE"] = 3.2671973387192588
        self.lr_bces_train["SR.MMP"] = 7.396136544059275
        self.lr_bces_train["SR.p53"] = 4.144670335841851

        #linear regression on test data
        self.lr_bces_test["NR.AhR"] = 12.78688524590164
        self.lr_bces_test["NR.AR"] = 2.3890784982935154
        self.lr_bces_test["NR.AR.LBD"] = 2.577319587628866
        self.lr_bces_test["NR.Aromatase"] = 7.575757575757576
        self.lr_bces_test["NR.ER"] = 12.015503875968992
        self.lr_bces_test["NR.ER.LBD"] = 4.5
        self.lr_bces_test["NR.PPAR.gamma"] = 6.2809917355371905
        self.lr_bces_test["SR.ARE"] = 19.45945945945946
        self.lr_bces_test["SR.ATAD5"] = 6.270096463022508
        self.lr_bces_test["SR.HSE"] = 5.245901639344262
        self.lr_bces_test["SR.MMP"] = 10.681399631675875
        self.lr_bces_test["SR.p53"] = 8.603896103896103

        #random forest on train data
        self.rf_bces_train["NR.AhR"] = 0.23693875133278047
        self.rf_bces_train["NR.AR"] = 0.145017609281127
        self.rf_bces_train["NR.AR.LBD"] = 0.03389064618165386
        self.rf_bces_train["NR.Aromatase"] = 0.12096774193548387
        self.rf_bces_train["NR.ER"] = 0.33911077618688773
        self.rf_bces_train["NR.ER.LBD"] = 0.15486725663716813
        self.rf_bces_train["NR.PPAR.gamma"] = 0.04733167672464797
        self.rf_bces_train["SR.ARE"] = 0.18916362653695445
        self.rf_bces_train["SR.ATAD5"] = 0.07476236249065471
        self.rf_bces_train["SR.HSE"] = 0.13068789354877033
        self.rf_bces_train["SR.MMP"] = 0.21169621593014024
        self.rf_bces_train["SR.p53"] = 0.056160844659103676

        #random forest on test data
        self.rf_bces_test["NR.AhR"] = 9.672131147540984
        self.rf_bces_test["NR.AR"] = 1.8771331058020477
        self.rf_bces_test["NR.AR.LBD"] = 1.7182130584192439
        self.rf_bces_test["NR.Aromatase"] = 7.196969696969697
        self.rf_bces_test["NR.ER"] = 8.13953488372093
        self.rf_bces_test["NR.ER.LBD"] = 3.1666666666666665
        self.rf_bces_test["NR.PPAR.gamma"] = 5.12396694214876
        self.rf_bces_test["SR.ARE"] = 14.594594594594595
        self.rf_bces_test["SR.ATAD5"] = 5.787781350482315
        self.rf_bces_test["SR.HSE"] = 3.6065573770491803
        self.rf_bces_test["SR.MMP"] = 9.94475138121547
        self.rf_bces_test["SR.p53"] = 6.6558441558441555



    # 'Gets sample at ID index. just index in input/output array'
    # def __getitem__(self, index):
    #     return self.X[index], self.y[index]
