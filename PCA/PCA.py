import Data
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

import pdb

import pickle

#https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

print("loadign data")
dataset = Data.Dataset()
m = 5070 #m samples
n = 229432 #n features
num_features = m

print("Starting pca")
pca = PCA(n_components=1)
new_output = pca.fit_transform(dataset.y)

print("Saving new output")

np.save("PCA1Sklearn.npy", new_output)

pdb.set_trace()

#pickle.dump(pca, open("PCA-Object.pickle", "wb"))

# evals = np.load(open("PCASklearn.npy", 'rb'))
# accum_evals = np.cumsum(evals)

# plt.step(range(len(accum_evals)), accum_evals*100)
# plt.title("Accumulated percent variance of each feature/component (log scale)")
# plt.xlabel("Feature/component number [1-5070]")
# plt.ylabel("Accumulated percent variance")
# plt.xscale("log")
# #plt.ylim([0, 1])
# plt.savefig("PCASklearn.png")



