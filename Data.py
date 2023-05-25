# import input_output
#https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
import numpy as np
import torch
import pickle
from data_assembly import Normalizer
import pdb

class Dataset:

    #so ik the websit told me to use a different format but I'm not motivated enough to change my current format yet
    def __init__(self):
        self.X = pickle.load(open("data_assembly/normalizedX.npy", "rb"))
        self.y = pickle.load(open("data_assembly/normalizedY.npy", "rb"))

    'Returns total numper of samples'
    def __len__(self):
        return len(self.X)

    'Gets sample at ID index. just index in input/output array'
    def __getitem__(self, index):
        return self.X[index], self.y[index]
