from xmlrpc.client import Boolean
from .dynamic import DynamicDataset
import numpy as np
import torch
from pyexlab.utils import get_info

#Returns two points in a metric space and their metric distance
class MetricSpaceDataset(DynamicDataset):

    def __init__(self, metric, space, precompute = None, subspace = None):

        #If precompute is a Boolean, then invoke the method "preCompute"
        #Otherwise assume that preCompute is an instance of the precomputation
        self.info_dict = {}
        self.info_dict['subspace'] = subspace

        self.metric = metric
        if metric is not None:
            self.info_dict['Metric'] = metric.__name__
        else:
            self.info_dict['Metric'] = metric

        if subspace is not None:
            self.space = space[subspace]
        else:
            self.space = space

        if precompute is not None:
            self.precompute = True
            if type(precompute) == Boolean and precompute == True:
                self.preCompute()
            elif type(precompute) != Boolean:
                self.H = precompute
            else:
                self.precompute = False
        
        self.info_dict['precomputed'] = self.precompute
        self.n = len(self.space)
        self.info_dict['len'] = self.n
        

    def __len__(self):

        return self.n**2

    def __getitem__(self, idx):

        i = int(idx / self.n)
        j = int(idx % self.n)

        if self.precompute:
            return [ [self.space[i], self.space[j]], torch.tensor([self.H[i,j]], dtype=torch.float32)]
        else:
            return [ [self.space[i], self.space[j]], torch.tensor([self.metric(self.space[i], self.space[j])], dtype=torch.float32)]

    def preCompute(self):

        self.H = self.metric(self.space.numpy())
        return torch.from_numpy(self.H)

    def getPreCompute(self):
        return self.H

    def info(self):
        return self.info_dict