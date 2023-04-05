import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

#using https://towardsdatascience.com/pytorch-tabular-regression-428e9c9ac93
#official pytorch classification https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
class ToxicityRegressor(nn.Module):
    def __init__(self, dim_input, dim_output):
        super(ToxicityRegressor, self).__init__()

        self.layer_1 = nn.Linear(dim_input, 450)
       # self.layer_2 = nn.Linear(450, 450)
        self.layer_3 = nn.Linear(450, 450)
        self.layer_out = nn.Linear(450, dim_output)
        
        self.n_layers = 3
        self.linears = [self.layer_1, self.layer_3, self.layer_out]
        self.use_bilinear = False

        self.relu = nn.ReLU()

    def forward(self, input):

        x = self.relu(self.layer_1(input))
      #  x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.layer_out(x)

        return x

    def predict(self, test_inputs):

        x = self.relu(self.layer_1(test_inputs))
       # x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.layer_out(x)

        return x
    
    def reset_parameters(self):
        self.layer_1.reset_parameters()
       # self.layer_2.reset_parameters()
        self.layer_3.reset_parameters()
        self.layer_out.reset_parameters()
    

    def compute_grad_norm(self):
        '''Compute the norm of the gradients per layer.'''  

        grad_norm = {}  

        for l in range(self.n_layers): 
            layer_norm = 0  
            layer_norm += self.linears[l].weight.grad.detach().data.norm(2).item()**2  
            layer_norm += self.linears[l].bias.grad.detach().data.norm(2).item()**2  

            if self.use_bilinear:  
                layer_norm += self.bilinears[l].weight.grad.detach().data.norm(2).item()**2  
                layer_norm += self.bilinears[l].bias.grad.detach().data.norm(2).item()**2  

            grad_norm[l] = layer_norm ** 0.5  

        return np.array(list(grad_norm.values()))  