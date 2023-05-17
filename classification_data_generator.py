import pandas as pd
import numpy as np
import pdb

df = pd.read_csv("assays/AID_1671196_p450-2d6-p1.csv", skiprows=[1, 2, 3, 4, 5])

outcomes = df["PUBCHEM_ACTIVITY_OUTCOME"]

outcomes_bin = []
for string in outcomes:
    if string == "Active":
        outcomes_bin.append(0)
    if string == "Inactive":
        outcomes_bin.append(1)
    if string == "Inconclusive":
        outcomes_bin.append(2)

pdb.set_trace()

np.save("outcomes.npy", np.array(outcomes_bin))