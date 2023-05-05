import pandas as pd
import numpy as np
import pdb
import math
import matplotlib.pyplot as plt

df = pd.read_csv("assays/AID_1159525_ap1-agonist-p1_viability.csv", skiprows=[1, 2, 3, 4, 5])

#ith replicate, jth molecule
activity_025 = []

for i in range(1, 52):
    arr = df["Fit_HillSlope-Replicate_" + str(i)]
    activity_025.append(arr)

activity_025 = np.array(activity_025)

#get std dev of each molecule
f = open("ree.txt", "a")
normalized_stds = []
num_nan = []
for molecule in activity_025.T:
    activities = np.array(molecule)
    normalized_stds.append(np.nanstd(activities) / np.nanmean(activities))
    f.write( str(np.nanstd(activities) / np.nanmean(activities)) + "\ne")
    #pdb.set_trace()
    num_nan.append(np.count_nonzero(np.isnan(activities)))


num_nan = np.array(num_nan)
normalized_stds = np.array(normalized_stds)

f.close()

pdb.set_trace()

print("Normalized std:", np.nanstd(activity_025.T[1]) / np.nanmean(activity_025.T[1]))
print("avg:", np.nanmean(activity_025[1]))

plt.bar([i for i in range (1, 52)], activity_025.T[1])
plt.xlabel("replicate")
plt.ylabel("Value")
plt.savefig("activity")

#pdb.set_trace()

# plt.hist(arr, bins=100)
# plt.xlabel("value")
# plt.ylabel("frequency")
# #plt.xscale("symlog")
# plt.savefig("hist")