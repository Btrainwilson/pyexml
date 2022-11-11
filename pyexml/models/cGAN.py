import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, hidden_channels = [256, 128, 64], kernel_size_list = [3, 4, 3, 4], stride_list = [2, 1, 2, 2], input_dim=1, im_chan=1):
      super(Generator, self).__init__()
      self.input_dim = input_dim
        
      #Initial layer
      self.h_layers = nn.ModuleList([nn.ConvTranspose2d(input_dim, hidden_channels[0], kernel_size=kernel_size_list[0], stride=stride_list[0])])
      self.h_layers.extend([nn.BatchNorm2d(hidden_channels[0])])
      self.h_layers.extend([nn.LeakyReLU(0.02, inplace=True)])
        
      #ReLU of hidden layers
      for i in range(len(hidden_channels) - 1):
          self.h_layers.extend([nn.ConvTranspose2d(hidden_channels[i], hidden_channels[i + 1], kernel_size=kernel_size_list[i+1], stride=stride_list[i+1])])
          self.h_layers.extend([nn.BatchNorm2d(hidden_channels[i + 1])])
          self.h_layers.extend([nn.LeakyReLU(0.02, inplace=True)])
                                       
      #Final layer TanH function
      self.h_layers.extend([nn.ConvTranspose2d(hidden_channels[-1], im_chan, kernel_size = kernel_size_list[-1], stride = stride_list[-1])])
      self.h_layers.extend([nn.Tanh()])                           
        
    def forward(self, noise):
      x = noise.view(len(noise), self.input_dim, 1, 1)
      for i in range(len(self.h_layers) - 1):
        x = self.h_layers[i](x)
      
      return self.h_layers[-1](x)

class Discriminator(nn.Module):
    def __init__(self, im_chan=1, hidden_dimensions = [64, 128], kernel_size = 4, stride = 2):
        super(Discriminator, self).__init__()
        self.im_chan = im_chan
        
        #First layer
        self.h_layers = nn.ModuleList([nn.Conv2d(im_chan, hidden_dimensions[0], kernel_size, stride)])
        self.h_layers.extend([nn.BatchNorm2d(hidden_dimensions[0])])
        self.h_layers.extend([nn.LeakyReLU(0.02, inplace=True)])
        
        #ReLU of hidden layers
        for i in range(len(hidden_dimensions) - 1):
            self.h_layers.extend([nn.Conv2d(hidden_dimensions[i], hidden_dimensions[i + 1], kernel_size, stride)])
            self.h_layers.extend([nn.BatchNorm2d(hidden_dimensions[i + 1])])
            self.h_layers.extend([nn.LeakyReLU(0.02, inplace=True)])
                                       
        #Final layer convolution
        self.h_layers.extend([nn.Conv2d(hidden_dimensions[-1], 1, kernel_size, stride)])
        
        
    def forward(self, image):
      x = image
      for i in range(len(self.h_layers) - 1):
        x = self.h_layers[i](x)
      disc_pred = self.h_layers[-1](x)
      return disc_pred.view(len(disc_pred), -1)
