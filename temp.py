import pandas as pd
from cmath import nan

cids_file = open("cids_strict.txt")
whole_thing = cids_file.read()
cids = whole_thing.split('\n')
cids_file.close()

assay_file = open("assays/AID_743036_ar-bla-agonist-p1.csv")
a_ids = pd.read_csv(assay_file, usecols=['PUBCHEM_CID'])['PUBCHEM_CID']
assay_file.close()

hash = {}
for id in a_ids:
    try:
        hash[str(int(id))] = 1
    except:
        pass

count = 0


output = open("test_cids.txt", "w")
for cid in cids:
    if cid in hash:
        output.write(cid + '\n')

