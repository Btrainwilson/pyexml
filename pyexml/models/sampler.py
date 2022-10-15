from .model import Model
from pyexlab.utils import get_info
import torch
from torch.autograd import Variable
import numpy as np

class Sampler():
    pass

class VariationalModel(Model):

    def __init__(self, model, sampler):
        super().__init__()
        pass

class GaussianSampleModel(Model):

    def __init__(self, mu_model, logvar_model, out_dim):
        super().__init__()
        self.mu_model = mu_model
        self.logvar_model = logvar_model
        self.out_dim = out_dim

    def forward(self, x):
        mu = self.mu_model(x)
        logvar = self.logvar_model(x)
        std = torch.exp(logvar / 2)
        sampled_z = Variable(torch.Tensor(np.random.normal(0, 1, (mu.size(0), self.out_dim))))
        z = sampled_z * std + mu
        return z
