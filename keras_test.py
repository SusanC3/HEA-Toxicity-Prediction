#for each strict id: input is stuff from dense train file, train on efficacy from assays/AID_743036_ar-bla-agonist-p1.csv

import pandas as pd
import csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import math
import numpy as np


print("Putting together hash")
ids_file = open("test_ids.txt")
whole_thing = ids_file.read()
id_pairs = whole_thing.split('\n')
#cid in, toxids out
ids_hash = {}
for i in range(len(id_pairs)):
    id_pairs[i] = id_pairs[i].split()
    if (len(id_pairs[i]) < 2):
        continue
    if id_pairs[i][0] not in ids_hash:
        ids_hash[id_pairs[i][0]] = []
    ids_hash[id_pairs[i][0]].append(id_pairs[i][1]) #keep in mind different


#put together list of outputs
print("Getting output")
output = []
cid_to_output = {}
output_file = open("assays/AID_743036_ar-bla-agonist-p1.csv")
read = pd.read_csv(output_file, usecols=['Efficacy-Replicate_1', 'PUBCHEM_CID'], dtype={'Efficacy-Replicate_1':str, 'PUBCHEM_CID':str})
cids = read['PUBCHEM_CID']
potency = read['Efficacy-Replicate_1']
for i in range(len(cids)):
    cid = cids[i]
    p = potency[i]
    if (type(cid) == str and type(p) == str and cid in ids_hash and cid not in cid_to_output):
        output.append(p)
        cid_to_output[cid] = p


#put together list of input vectors, containing all features from tox21 dense features
print("Getting input")
input = []
id_to_input = {}
input_file = open("tox21_dense_train.csv")
input_csv = csv.reader(input_file)
ids_in_input = {}
for row in input_csv:
    if (len(row[0]) > 0 and type(row[0]) == str):
        ids_in_input[row[0]] = row[1:]


not_added = []
for cid in cid_to_output:
    added = False
    for id in ids_hash[cid]:
        if id in ids_in_input:
            input.append(ids_in_input[id])
            id_to_input[id] = input[len(input)-1]
            added = True
            break
    if not added:
        not_added.append(cid)

for cid in not_added:
    output.remove(cid_to_output[cid])

print("Converting input to floats")
for sample in input:
    for i in range(len(sample)):
        sample[i] = float(sample[i])
        
print("Converting output to floats")
for i in range(len(output)):
    output[i] = float(output[i])


print("defining model")
def baseline_model():
    model = Sequential()
    model.add(Dense(801, input_shape=(801,), kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='RMSprop')
    return model

print("Starting to train")
estimator = KerasRegressor(model=baseline_model, epochs=100, batch_size = 64, verbose=0)
#use kfold for now, in the futre want to use whole data with just this test data as test
kfold = KFold(n_splits=10)
results = cross_val_score(estimator, input, output, cv=kfold, scoring='neg_mean_squared_error')
print('Baseline:', results.mean(), "(", results.std(), ") MSE")