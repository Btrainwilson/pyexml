
import torch
import torch.nn as nn
import torch.functional as F
from .ffwd import Simple_FFWDNet
from .model import Model
from pyexlab.utils import get_info

class MetricNet(Model):
    
    def __init__(self, base_model, metric, device = torch.device("cpu")):
        super().__init__()

        self.bnet = base_model
        self.l = torch.nn.Parameter(torch.ones(1,1, dtype=torch.float32))
        self.metric = metric

        self.info_dict['Map Net Info'] = get_info(self.bnet)
        
        self.device = device
        self.to(self.device)

    def forward(self, x):
        
        r1 = x[0].float()
        r2 = x[1].float()
        r1 = self.bnet(r1)
        r2 = self.bnet(r2)
        D = self.metric(r1 , r2)
        D = D * self.l 

        return torch.squeeze(D)

    def getBaseModel(self):
        return self.bnet
