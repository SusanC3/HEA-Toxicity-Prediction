import pandas as pd
import numpy as np
import pdb

df = pd.read_csv("assays/AID_1671196_p450-2d6-p1.csv", skiprows=[1, 2, 3, 4, 5])

activities = []

for i in range(1, 52): #for all 52 replicates
    arr = df["Activity_Score-Replicate_" + str(i)]
    activities.append(arr)
activities = np.array(activities)

scores = []

for molecule in activities.T:
    replicate_scores = np.array(molecule)
    mean_score = np.nanmean(replicate_scores)
    scores.append(mean_score)

scores = np.array(scores)

np.save("activity_scores.npy", scores)