import numpy as np
import pandas as pd
import pickle
import pdb

assay_names = open("assay_names.txt", "r").read().split()
potential_ids = pickle.load(open("selection/similar_tox21s.npy", "rb"))
tox21_to_smiles = pickle.load(open("selection/tox21_to_smiles_hash.obj", "rb"))
assay_df = pd.read_csv("assays/" + assay_names[0], usecols=["PUBCHEM_EXT_DATASOURCE_SMILES"], skiprows=[1, 2, 3, 4, 5])

#for now
print("assembling hash")
smiles_in_assay = {}
for smi in assay_df["PUBCHEM_EXT_DATASOURCE_SMILES"]:
    smiles_in_assay[smi] = 1

print("excluding absent ids")
toxids_to_use = []
for id in potential_ids:
    if tox21_to_smiles[id] in smiles_in_assay:
        toxids_to_use.append(id)

pickle.dump(np.array(toxids_to_use), open("tox21s_to_use.npy", "wb"))