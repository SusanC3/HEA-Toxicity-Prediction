import pandas as pd
import numpy as np
import pdb
import math
import matplotlib.pyplot as plt

df = pd.read_csv("assays/AID_1671196_p450-2d6-p1.csv", skiprows=[1, 2, 3, 4, 5])

f = open("dose_response.txt", "a")

#data for first molecule is going to be first element of each relevant column
for name in df.columns:
    if name[:9] == "Activity ": #and name[-2:] == "_1":
        dose = float(name[12:-15].strip())
        response = float(df[name][10])
        if not math.isnan(response):
            print(name)
            f.write(str(dose) + '\t' + str(response) + '\n')
f.close()