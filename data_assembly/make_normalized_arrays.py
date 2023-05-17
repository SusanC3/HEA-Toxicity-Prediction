import pickle
import Normalizer
import numpy as np
import torch
import pdb

data_hash = pickle.load(open("data_hash.obj", "rb"))

X = []
y = []

for id in data_hash:
    X.append(data_hash[id][0])
    y.append(data_hash[id][1])

input_normalizer = Normalizer.UnitGaussianNormalizer(torch.FloatTensor(X))
output_normalizer = Normalizer.UnitGaussianNormalizer(torch.FloatTensor(y)) 

normalizedX = input_normalizer.encode(torch.FloatTensor(X)).data.numpy()
normalizedy = output_normalizer.encode(torch.FloatTensor(y)).data.numpy()

pickle.dump(normalizedX, open("normalizedX.npy", "wb"))
pickle.dump(normalizedy, open("normalizedy.npy", "wb"))
