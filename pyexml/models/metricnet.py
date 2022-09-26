
import torch
import torch.nn as nn
import torch.functional as F
from .ffwd import Simple_FFWDNet
from pyexlab.utils import get_info

class MetricNet(nn.Module):
    __name__ = "MetricNet"
    def __init__(self, base_model, metric):
        super().__init__()

        self.info_dict = {}
        self.bnet = base_model
        self.l = torch.nn.Parameter(torch.ones(1,1, dtype=torch.float32))
        self.metric = metric

    def forward(self, x):
        
        r1 = x[0].float()
        r2 = x[1].float()
        r1 = self.bnet(r1)
        r2 = self.bnet(r2)
        D = self.metric(r1 , r2)
        D = D * self.l 

        return D.T

    def getBaseModel(self):
        return self.bnet

    def info(self):
        self.info_dict['Name'] = self.__name__
        self.info_dict['map_net info'] = get_info(self.bnet)
        self.info_dict['map_net'] = self.bnet
        self.info_dict['Metric'] = self.metric.__name__
        self.info_dict['state_dict'] = self.state_dict()
        return self.info_dict
