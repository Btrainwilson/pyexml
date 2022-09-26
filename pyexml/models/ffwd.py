#Feedforward Neural Net
import torch.nn as nn
import torch.nn.functional as F


class Simple_FFWDNet(nn.Module):
    __name__ = "Simple_FFWDNet"

    def __init__(self, bit_num, real_num, layers=5, latent_size=50):
        super(Simple_FFWDNet, self).__init__()

        self.bit_num = bit_num
        self.real_num = real_num
        self.layers = layers
        self.latent_size = latent_size
        
        self.h_layers = nn.ModuleList([nn.Linear(bit_num, latent_size)])
        for i in range(layers - 2):
            self.h_layers.extend([nn.Linear(latent_size, latent_size)])

        self.h_layers.extend([nn.Linear(latent_size, real_num)])

        self.info_dict = {}
        self.info_dict['Name'] = __name__

        for key in self.__dict__.keys():
            self.info_dict[key] = self.__dict__[key]

        
    def forward(self, x):
        
        for i in range(len(self.h_layers) - 1):
            x = F.relu(self.h_layers[i](x))

        x = self.h_layers[-1](x)

        return x

    def info(self):
        return self.info_dict