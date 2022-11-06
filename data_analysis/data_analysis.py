import pandas as pd

print("reading files")

cids_file = open("cids_full.txt")
whole_thing = cids_file.read()
cids = whole_thing.split('\n')
cids_file.close()

cancer_file = open("cancer.txt")
whole_thing = cancer_file.read()
cancer_assay_names = whole_thing.split('\n')
cancer_file.close()

kidney_file = open("kidneytox.txt")
whole_thing = kidney_file.read()
kidney_assay_names = whole_thing.split('\n')
kidney_file.close()

liver_file = open("livertox.txt")
whole_thing = liver_file.read()
liver_assay_names = whole_thing.split('\n')
liver_file.close()

met_file = open("metabolism.txt")
whole_thing = met_file.read()
met_assay_names = whole_thing.split('\n')
met_file.close()

cancer_hash = {}
kidney_hash = {}
liver_hash = {}
met_hash = {}

print("processing cancer assays")

for name in cancer_assay_names:
    a_file = open("assays/" + name, "r")
    #print(name)
    a_read = pd.read_csv(a_file, usecols=['Efficacy-Replicate_1', 'PUBCHEM_CID'], dtype={'Efficacy-Replicate_1':str, 'PUBCHEM_CID':str})
    a_cids = a_read['PUBCHEM_CID']
    a_potency = a_read['Efficacy-Replicate_1']
    a_file.close()

    if len(a_cids) != len(a_potency):
        print("oh no")

    for i in range(len(a_cids)):
        if type(a_cids[i]) != str or type(a_potency[i]) != str:
            continue
        cancer_hash[a_cids[i]] = a_potency[i]

print("processing kidney assays")

for name in kidney_assay_names:
    a_file = open("assays/" + name, "r")
    #print(name)
    try:
        a_read = pd.read_csv(a_file, usecols=['Efficacy-Replicate_1', 'PUBCHEM_CID'], dtype={'Efficacy-Replicate_1':str, 'PUBCHEM_CID':str})
    except:
        print("boo")
    a_cids = a_read['PUBCHEM_CID']
    a_potency = a_read['Efficacy-Replicate_1']
    a_file.close()

    if len(a_cids) != len(a_potency):
        print("oh no")

    for i in range(len(a_cids)):
        if type(a_cids[i]) != str or type(a_potency[i]) != str:
            continue
        kidney_hash[a_cids[i]] = a_potency[i]

print("processing liver assays")

for name in liver_assay_names:
    a_file = open("assays/" + name, "r")
    #print(name)
    a_read = pd.read_csv(a_file, usecols=['Efficacy-Replicate_1', 'PUBCHEM_CID'], dtype={'Efficacy-Replicate_1':str, 'PUBCHEM_CID':str})
    a_cids = a_read['PUBCHEM_CID']
    a_potency = a_read['Efficacy-Replicate_1']
    a_file.close()

    if len(a_cids) != len(a_potency):
        print("oh no")

    for i in range(len(a_cids)):
        if type(a_cids[i]) != str or type(a_potency[i]) != str:
            continue
        liver_hash[a_cids[i]] = a_potency[i]

print("processing metabolism assays")

for name in met_assay_names:
    a_file = open("assays/" + name, "r")
   # print(name)
    a_read = pd.read_csv(a_file, usecols=['Efficacy-Replicate_1', 'PUBCHEM_CID'], dtype={'Efficacy-Replicate_1':str, 'PUBCHEM_CID':str})
    a_cids = a_read['PUBCHEM_CID']
    a_potency = a_read['Efficacy-Replicate_1']
    a_file.close()

    if len(a_cids) != len(a_potency):
        print("oh no")

    for i in range(len(a_cids)):
        if type(a_cids[i]) != str or type(a_potency[i]) != str:
            continue
        met_hash[a_cids[i]] = a_potency[i]

cancer_count = 0
kidney_count = 0
liver_count = 0
met_count = 0

print("processing counts")

for cid in cids:
    if cid in cancer_hash:
        cancer_count += 1
    if cid in kidney_hash:
        kidney_count += 1
    if cid in liver_hash:
        liver_count += 1
    if cid in met_hash:
        met_count += 1

print("cancer count:", cancer_count)
print("kidney count:", kidney_count)
print("liver count:", liver_count)
print("met count:", met_count)