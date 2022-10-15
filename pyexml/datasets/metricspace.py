from xmlrpc.client import Boolean
from .dynamic import DynamicDataset
import numpy as np
import torch
from pyexlab.utils import get_info

#Returns two points in a metric space and their metric distance
class MetricSpaceDataset(DynamicDataset):

    def __init__(self, metric, space, precompute = None, subspace = None, device = torch.device("cpu"), dtype = torch.float32):

        #If precompute is a Boolean, then invoke the method "preCompute"
        #Otherwise assume that preCompute is an instance of the precomputation


        self.metric = metric
        self.device = device
        self.dtype = dtype
        
        if subspace is not None:
            self.space = torch.from_numpy(space[subspace])
        else:
            self.space = torch.from_numpy(space)

        self.space = self.space.to(device=self.device, dtype=self.dtype)

        self.n = len(self.space)

        if precompute is not None:
            self.precompute = True
            if type(precompute) == Boolean and precompute == True:
                self.preCompute()
            elif type(precompute) != Boolean:
                if subspace is not None:
                    sub_idx = tuple(np.squeeze(np.meshgrid(subspace, subspace, indexing='ij')))
                    self.H = torch.from_numpy(np.squeeze(precompute[sub_idx]))
                    
                elif type(precompute) == torch.Tensor:
                    self.H = precompute
                else:
                    self.H = torch.from_numpy(precompute)

                self.H = self.H.to(device=self.device, dtype=self.dtype)
            else:
                self.precompute = False

        

        self.info_dict = {}
        self.info_dict['Subspace'] = subspace

        if metric is not None:
            self.info_dict['Metric'] = metric.__name__
        else:
            self.info_dict['Metric'] = "None"

        self.info_dict['Precomputed'] = self.precompute
        self.info_dict['Length'] = self.n
        

    def __len__(self):

        return self.n**2

    def __getitem__(self, idx):

        i = int(idx / self.n)
        j = int(idx % self.n)

        if self.precompute:
            return [ [self.space[i], self.space[j]], self.H[i,j]]
        else:
            h = torch.from_numpy(self.metric(self.space[i].numpy(), self.space[j].numpy()))
            h = h.to(device=self.device, dtype = self.dtype)
            
            return [ [self.space[i], self.space[j]], h]

    def preCompute(self):

        self.H = torch.tensor(self.metric(self.space.numpy()), device=self.device, dtype=self.dtype)
        return torch.from_numpy(self.H)

    def getPreCompute(self):
        return self.H

    def info(self):
        return self.info_dict