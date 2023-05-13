# import input_output
#https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
import numpy as np
import torch
import pdb

class Dataset:

    #so ik the websit told me to use a different format but I'm not motivated enough to change my current format yet
    def __init__(self):
        # self.labels = labels
        # self.list_IDS = list_IDs
        #self.X, self.y = input_output.get_input_output()
        self.Xraw = np.load(open("input.npy", "rb"), allow_pickle=True)
       # self.Xraw = np.load(open("PCA/PCA100Input.npy", "rb"), allow_pickle=True)
        #self.yraw = np.load(open("output.npy", "rb"), allow_pickle=True)
        self.yraw = np.load(open("activity_scores.npy", "rb"), allow_pickle=True)
      #  self.yraw = np.load(open("PCA/PCA1Sklearn.npy", "rb"), allow_pickle=True)

        self.input_normalizer, self.output_normalizer = UnitGaussianNormalizer(torch.from_numpy(self.Xraw)), UnitGaussianNormalizer(torch.from_numpy(self.yraw))

        self.X = self.input_normalizer.encode(torch.from_numpy(self.Xraw)).data.numpy()
        self.y = self.output_normalizer.encode(torch.from_numpy(self.yraw)).data.numpy()

    'Returns total numper of samples'
    def __len__(self):
        return len(self.X)

    'Gets sample at ID index. just index in input/output array'
    def __getitem__(self, index):
        return self.X[index], self.y[index]


#Here is some code to get you started on normalization  
class UnitGaussianNormalizer(object):  
    def __init__(self, x, eps=0.00001):  
        super(UnitGaussianNormalizer, self).__init__()  
  
        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T  
        self.mean = torch.mean(x, 0)  
        self.std = torch.std(x, 0)  
        self.eps = eps  
  
    def encode(self, x):  
       # pdb.set_trace()
        x = (x - self.mean) / (self.std + self.eps)  
        return x  
  
    def decode(self, x):  
        std = self.std + self.eps # n  
        mean = self.mean  
  
        # x is in shape of batch*n or T*batch*n  
        x = (x * std) + mean  
        return x  
  
    def cuda(self):  
        self.mean = self.mean.cuda()  
        self.std = self.std.cuda()  
  
    def cpu(self):  
        self.mean = self.mean.cpu()  
        self.std = self.std.cpu()

