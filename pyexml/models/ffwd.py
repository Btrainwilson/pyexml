#Feedforward Neural Net
import torch.nn as nn
import torch.nn.functional as F
from .model import Model
import torch

class Simple_FFWDNet(Model):

    def __init__(self, bit_num, real_num, layers=5, latent_size=50, device = torch.device("cpu")):
        super().__init__()
        self.bit_num = bit_num
        self.real_num = real_num
        self.layers = layers
        self.latent_size = latent_size
        
        self.h_layers = nn.ModuleList([nn.Linear(bit_num, latent_size)])
        for i in range(layers - 2):
            self.h_layers.extend([nn.Linear(latent_size, latent_size)])

        self.h_layers.extend([nn.Linear(latent_size, real_num)])

        self.device = device
        self.to(self.device)
        
    def forward(self, x):
        
        for i in range(len(self.h_layers) - 1):
            x = F.relu(self.h_layers[i](x))

        x = self.h_layers[-1](x)

        return x