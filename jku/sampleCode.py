# make sure  numpy, scipy, pandas, sklearn are installed, otherwise run
# pip install numpy scipy pandas scikit-learn
import numpy as np
import pandas as pd
from scipy import io
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import pdb

from jku_Data import Dataset
dataset = Dataset()

# load data

y_tr = pd.read_csv('tox21_labels_train.csv', index_col=0)
y_te = pd.read_csv('tox21_labels_test.csv', index_col=0)
x_tr_dense = pd.read_csv('tox21_dense_train.csv', index_col=0).values
x_te_dense = pd.read_csv('tox21_dense_test.csv', index_col=0).values
# x_tr_sparse = io.mmread('tox21_sparse_train.mtx.gz').tocsc()
# x_te_sparse = io.mmread('tox21_sparse_test.mtx.gz').tocsc()

# filter out very sparse features
# sparse_col_idx = ((x_tr_sparse > 0).mean(0) > 0.05).A.ravel()
# x_tr = np.hstack([x_tr_dense, x_tr_sparse[:, sparse_col_idx].A])
# x_te = np.hstack([x_te_dense, x_te_sparse[:, sparse_col_idx].A])

#what if if I try with no sparse features
x_tr = x_tr_dense
x_te = x_te_dense

# Build a random forest model for all twelve assays
for target in y_tr.columns:
    rows_tr = np.isfinite(y_tr[target]).values #stores true/false is data NaN
    rows_te = np.isfinite(y_te[target]).values #stores true/false is data NaN
    
    # dataset.X_train[target] == x_tr[rows_tr]
    # dataset.y_train[target] == y_tr[target][rows_tr]
    # dataset.X_test[target] == x_te[rows_te]
    # dataset.y_test[target] == y_te[target][rows_te]

    # print(np.where(dataset.X_train[target] != x_tr[rows_tr]))
    # print(np.where(dataset.y_train[target] != y_tr[target][rows_tr]))
    # print(np.where(dataset.X_test[target] != x_te[rows_te]))
    # print(np.where(dataset.y_test[target] != y_te[target][rows_te]))

    #so data values are the same
    
    rf = RandomForestClassifier(n_estimators=100,  n_jobs=4, random_state=0)
    
    #with jku data
    rf.fit(x_tr[rows_tr], y_tr[target][rows_tr])
    p_te1 = rf.predict_proba(x_te[rows_te])
    auc_te1 = roc_auc_score(y_te[target][rows_te], p_te1[:, 1])
    print("Jku:", target, auc_te1)

    #with my data
    rf.fit(dataset.X_train[target], dataset.y_train[target])
    p_te2 = rf.predict_proba(dataset.X_test[target])
    auc_te2 = roc_auc_score(dataset.y_test[target], p_te2[:, 1])
    print("Me :", target, auc_te2)
    print(auc_te1 == auc_te2)
    print()

  #  pdb.set_trace()

  #  print("%15s: %3.5f" % (target, auc_te))

  #but the predictions are not the same