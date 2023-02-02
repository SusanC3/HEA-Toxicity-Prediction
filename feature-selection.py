from sklearn.ensemble import RandomForestClassifier
import numpy as np

import time

import input_output
import bilateral_filter

#full fit will likely take around 2 hours
#ALSO gets killed because of memory before full fit completes

# X, Y = input_output.get_input_output()
# X, Y = bilateral_filter.filter(X, Y)

t = time.time()

#for testing fit feasibility with shape
X = np.zeros((5070, 801))
Y = np.ones((5070, 22943))  #final shape is 5070, 229432

print("making random forest")
random_forest = RandomForestClassifier(n_estimators=100, criterion='gini', min_samples_split=30, n_jobs=-1)

print("fitting random forest")
random_forest.fit(X, Y)

feature_importances = random_forest.feature_importances_
print(feature_importances)

print(time.time() - t)

