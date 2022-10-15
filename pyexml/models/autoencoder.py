import torch.nn as nn
import torch
import numpy as np
from .model import Model
from pyexlab.utils import get_info
from .ffwd import Simple_FFWDNet
from torch.autograd import Variable

class AutoEncoder(Model):

    def __init__(self, encoder, decoder):

        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.info_dict['Encoder Info'] = get_info(self.encoder)
        self.info_dict['Decoder Info'] = get_info(self.decoder)

    def forward(self, x):
        
        x = self.encoder(x)
        x = self.decoder(x)

        return x

class VariatonalEncoder(Model):

    def __init__(self, encoder, sampler):
        super().__init__()
        self.encoder = encoder
        self.sampler = sampler
        self.info_dict['Encoder Info'] = get_info(self.encoder)
        self.info_dict['Sampler Info'] = get_info(self.sampler)

    def forward(self, x):

        x = self.encoder(x)
        z = self.sampler(x)

        return z
