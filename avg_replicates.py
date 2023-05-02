import numpy as np
import pandas as pd
import pdb

# columns 0-7 aren't data, cols 8-48 are replicate 1
# first col of each replicate is string "phenotype", maybe can try to predict later but can ignore for now

df = pd.read_csv("assays/AID_1671196_p450-2d6-p1.csv", skiprows=[1, 2, 3, 4])

#excluding: phenotype (8), analysis comment (11), curve description (13), excluded points (20) because i don't want strings rn
#also excluding compound qc because it's a string and it only occurs in replicate 1
names = df.columns.values.tolist()
datas = np.array(names[9:11] + [names[12]] + names[14:20] + names[21:48])

avgd_df = {}

for measurement in datas:
    cols = []
    for j in range(1, 52): #for each replicate
        col = np.array(df[measurement[:-1] + str(j)]) #len 9668
        cols.append(col)
    cols = np.array(cols)
    avgd_df[measurement[:-2]] = np.nanmean(cols, axis=0)

pdb.set_trace()