import pickle
import numpy as np
import pandas as pd
import pdb


#{id : [input, output]}
tox21s = pickle.load(open("tox21s_to_use.npy", "rb"))
smiles = pickle.load(open("selection/tox21_to_smiles_hash.obj", "rb"))

#assemble id : input hash
tox21_to_input = {}
input_df = pd.read_csv("tox21_dense_train.csv")
for index, row in input_df.iterrows():
    tox21_to_input[row[0]] = np.array(row[1:])

#assemble smiles : output hash
#only doing replicate 1 for now to make it easier
smiles_to_output = {}
output_df = pd.read_csv("assays/AID_1671196_p450-2d6-p1.csv", usecols=["PUBCHEM_EXT_DATASOURCE_SMILES", "Activity_Score-Replicate_1"], skiprows=[1, 2, 3, 4, 5])
for index, row in output_df.iterrows():
    smiles_to_output[row[0]] = row[1]

#combine
tox21_to_data = {}
for id in tox21s:
    tox21_to_data[id] = [tox21_to_input[id] , smiles_to_output[smiles[id]]]

pickle.dump(tox21_to_data, open("data_hash.obj", "wb"))