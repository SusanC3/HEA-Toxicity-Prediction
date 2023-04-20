from sklearn.linear_model import Ridge
import numpy as np
import Data
import pdb

dataset = Data.Dataset()

#will fit to first 90% of data, score on last 10%
X_fit = dataset.X[:int(0.9*len(dataset.X))]
y_fit = dataset.y[:int(0.9*len(dataset.y))]

X_score = dataset.X[int(0.9*len(dataset.X)):]
y_score = dataset.y[int(0.9*len(dataset.y)):]

model = Ridge()
model.fit(X_fit, y_fit)

preds_fit = model.predict(X_fit)
tot_squared_error = np.sum((preds_fit - y_fit)**2)
denominator = preds_fit.shape[0]*preds_fit.shape[1]
print("MSE fit:", tot_squared_error/denominator)

preds_score = model.predict(X_score)
tot_squared_error = np.sum((preds_score - y_score)**2)
denominator = preds_score.shape[0]*preds_score.shape[1]
print("MSE score:", tot_squared_error/denominator)