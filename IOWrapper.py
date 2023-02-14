# import input_output
import numpy as np

class Dataset:

    #so ik the websit told me to use a different format but I'm not motivated enough to change my current format yet
    def __init__(self):
        # self.labels = labels
        # self.list_IDS = list_IDs
        #self.X, self.y = input_output.get_input_output()
        self.X = np.load(open("input.npy", "rb"), allow_pickle=True)
        self.y = np.load(open("output.npy", "rb"), allow_pickle=True)

    'Returns total numper of samples'
    def __len__(self):
        return len(self.X)

    'Gets sample at ID index. just index in input/output array'
    def __getitem__(self, index):
        return self.X[index], self.y[index]