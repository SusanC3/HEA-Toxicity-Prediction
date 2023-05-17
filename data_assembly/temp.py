import pandas as pd
import pdb
import pickle

# df = pd.read_csv("tox21_dense_train.csv")
# tox21_ids = df[df.columns[0]]

# f = open("all_tox21_ids.txt", "a")
# for id in tox21_ids:
#     f.write(id + "\n")
# f.close()


tox21_to_smiles = {}

f = open("tox21_to_smiles.txt", "r")
count = 0
for line in f:
    if len(line.split()) < 2:
        count += 1
        continue #if we don't have a smiles for it we're not including it
    tox21, smiles = line.split()[0], line.split()[1]
    tox21_to_smiles[tox21] = smiles


print(count, "ids without smiles")

pickle.dump(tox21_to_smiles, open("tox21_to_smiles_hash.obj", "wb"))