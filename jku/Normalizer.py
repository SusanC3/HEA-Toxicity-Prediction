import torch

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

