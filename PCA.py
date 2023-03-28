import Data
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

import pdb

#https://pytorch.org/docs/stable/generated/torch.pca_lowrank.html#torch.pca_lowrank

print("loadign data")
dataset = Data.Dataset()
m = 5070 #m samples
n = 229432 #n features
num_features = m

print("Starting pca")
#U, S, V = torch.pca_lowrank(torch.from_numpy(dataset.y), q=num_features)
pca = PCA(n_components=m)
pca.fit(dataset.y)

print("making plot")
#evals = S**2/(m-1)

x = [i for i in range(num_features)]

plt.bar(x, pca.explained_variance_ratio_[:num_features])
plt.ylim([0, 1])
plt.savefig("PCASklearn.png")

pdb.set_trace()

np.save("PCASklearn.npy", pca.explained_variance_ratio_[:num_features])

# for i in range(num_features):
#     print(str(i) + "\t"+ str(evals[i].item()))

# shortened = np.matmul(torch.from_numpy(dataset.y), V[:, :num_features])

# pdb.set_trace()

# np.save("y500.npy", shortened.numpy())

