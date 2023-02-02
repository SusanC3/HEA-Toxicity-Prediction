from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import input_output

X, Y = input_output.get_input_output()

print("defining model")
def get_model():
    model = Sequential()
    model.add(Dense(5070, input_shape=(5070,), kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


print("standardizing data")
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(model=get_model, epochs=50, batch_size = 5, verbose=0)))
pipeline = Pipeline(estimators)
print("Starting training & assessment")
#use kfold for now, in the futre want to use whole data with just this test data as test
kfold = KFold(n_splits=10)
print("After kfold thing")
results = cross_val_score(pipeline, X, Y, cv=kfold, scoring='neg_mean_squared_error')
print('Results:', results.mean(), "(", results.std(), ") MSE")
#TODO WILL NEED TO RE-SCALE OUTPUT FOR ACTUALLY CORRECT GUESSES