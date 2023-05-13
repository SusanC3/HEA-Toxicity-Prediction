from cmath import nan
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import pdb

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


# pubchem = open("chemicals_HEA_substructure.csv")
# read = pd.read_csv(pubchem, usecols=['cid'])
# cids = read['cid']
# pubchem.close()

# print("reading init files")

# ms_file = open("model_smiles.txt")
# whole_thing = ms_file.read()
# model_smiles = whole_thing.split('\n')
# ms_file.close()

# assay_names_file = open("assays/lsoutput.txt")
# whole_thing = assay_names_file.read()
# assay_names = whole_thing.split('\n')
# assay_names_file.close()

count = 0
i = 0

print(tanimoto_calc("CC(C)C1=NC(=CS1)CN(C)C(=O)NC(C(C)C)C(=O)NC(CC2=CC=CC=C2)CC(C(CC3=CC=CC=C3)NC(=O)OCC4=CN=CS4)O", 
                    "CCCC1(CC(=C(C(=O)O1)C(CC)C2=CC(=CC=C2)NS(=O)(=O)C3=NC=C(C=C3)C(F)(F)F)O)CCC4=CC=CC=C4"))

pdb.set_trace()


hash = {}

print("assembling hash")
for name in assay_names:
    i += 1
    print("processing assay", i, "out of", len(assay_names))
    a_file = open("assays/" + name, "r")
    a_smiles = pd.read_csv(a_file, usecols=['PUBCHEM_EXT_DATASOURCE_SMILES'])['PUBCHEM_EXT_DATASOURCE_SMILES']
    a_file.close()

    for smi1 in a_smiles:
        if smi1 not in hash:
            hash[smi1] = 0

print("hash size:", len(hash.keys()))
print("checking for similarity")
for smi1 in hash.keys():
    if type(smi1) != str:
            continue
    for smi2 in model_smiles:
            t = tanimoto_calc(smi1, smi2)
            # print(t)
            if t > 0.15:
                hash[smi1] += 1
               # print("found a smilarity")

f = open("smiles.txt", "w")
for smi in hash:
    if hash[smi] > 0:
        count += 1
        f.write(str(smi) + ": " + str(hash[smi]) + "\n")



# for name in assay_names:
#     i += 1
#     j = 0
#     print("processing assay", i, "out of", len(assay_names))
#     a_file = open("assays/" + name, "r")
#     a_smiles = pd.read_csv(a_file, usecols=['PUBCHEM_EXT_DATASOURCE_SMILES'])['PUBCHEM_EXT_DATASOURCE_SMILES']
#     a_file.close()

#     for smi1 in a_smiles:
#         j += 1
#         # print("processing chemical", j, "out of", len(a_smiles))
#         if type(smi1) != str:
#             continue
#         if smi1 not in hash:
#             hash[smi1] = 0
#         for smi2 in model_smiles:
#             t = tanimoto_calc(smi1, smi2)
#             # print(t)
#             if t > 0.5:
#                 hash[smi1] += 1
#                 print("found a smilarity")

# for smi in hash:
#     if hash[smi] > 0:
#         count += 1
#         f.write(str(smi) + ": " + str(hash[smi]) + "\n")


#first: hash ids contained in assays, see how many sample ids are in those assays
# hash = {}
# for id in cids:
#     hash[id] = 0

# print(392622 in hash)

# for name in assay_names:
#     i += 1
#     print("processing assay", i, "out of", len(assay_names))
#     a_file = open("assays/" + name, "r")
#     a_cids = pd.read_csv(a_file, usecols=['PUBCHEM_CID'])['PUBCHEM_CID']
#     a_file.close()

#     for id in a_cids:
#         if id in hash:
#             hash[id] += 1

# for id in hash:
#     if hash[id] > 0:
#         count += 1
#         f.write(str(id) + ": " + str(hash[id]) + "\n")

f.close()
print('found', count, 'cids')

