from cmath import nan
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import pdb
import pickle
import numpy as np

def tanimoto_calc(smi1, smi2):
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    try:
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 3, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 3, nBits=2048)
        s = round(DataStructs.TanimotoSimilarity(fp1,fp2),3)
        return s
    except:
        print("error")
        return -1


tox21_to_smiles = pickle.load(open("tox21_to_smiles_hash.obj", "rb"))
model_smiles = open("model_smiles.txt", "r").read().split()
similarity_threshold = 0.15

similar_tox21s = {}

print("checking for similarity")
for tox21 in tox21_to_smiles.keys():
    smi1 = tox21_to_smiles[tox21]
    for smi2 in model_smiles:
        if (tanimoto_calc(smi1, smi2) > similarity_threshold):
            similar_tox21s[tox21] = 1
            break


array = np.array(list(similar_tox21s.keys()))
print('found', len(array), 'cids')

pickle.dump(array, open("similar_tox21s.npy", "wb"))


