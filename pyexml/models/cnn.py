from .model import Model
import torch.nn as nn
import torch.nn.functional as F

class DeepCNN(Model):

    def __init__(self, in_channels, layers):
        super().__init__()

        self.h_layers = nn.ModuleList([nn.Conv2d(in_channels, 32, (3,3))])
        self.h_layers.extend([nn.BatchNorm2d()])
        self.h_layers.extend([nn.MaxPool2d((3,3))])
        self.h_layers.extend([nn.Dropout(0.25)])
        
        for i in range(layers - 2):
            self.h_layers.extend([])

        self.conv2 = nn.Conv2d(6, 16, 5)




    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        x = self.batch_norm1(x)
        x = self.max_pool1(x)

