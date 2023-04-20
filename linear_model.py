from sklearn.linear_model import Ridge
import numpy as np
import Data
import pdb

dataset = Data.Dataset()

#will fit to first 90% of data, score on last 10%
X_train = dataset.X[:int(0.9*len(dataset.X))]
y_train = dataset.y[:int(0.9*len(dataset.y))]

X_test = dataset.X[int(0.9*len(dataset.X)):]
y_test = dataset.y[int(0.9*len(dataset.y)):]

model = Ridge()
model.fit(X_test, y_test)

preds_fit = model.predict(X_train)
tot_squared_error = np.sum((preds_fit - y_train)**2)
denominator = preds_fit.shape[0]*preds_fit.shape[1]
print("MSE train:", tot_squared_error/denominator)

preds_score = model.predict(X_test)
tot_squared_error = np.sum((preds_score - y_test)**2)
denominator = preds_score.shape[0]*preds_score.shape[1]
print("MSE test:", tot_squared_error/denominator)