import torch
from ..utilities import get_cuda
from ..backend import device

class Model(torch.nn.Module):
    """
    Wrapper class for a PyTorch model that includes additional information and loss computation functionality. 
    """
    def __init__(self):
        super(Model, self).__init__()

        #Default datatype
        self.dtype = torch.float32

        #Get best device
        self.to(device)

        self.info_dict = {}
        self.info_dict['Name'] = str(type(self))
        self.info_dict['state_dict'] = self.state_dict()
    
    def info(self):
        """
        Returns a dictionary containing information about the model and its state.
        """
        self.info_dict.update(self.__dict__)
        return {self.info_dict['Name'] : self.info_dict}

    def loss(self):
        """
        Computes loss for the model. 
        """
        loss_dict = {}
        loss_dict['Total'] = 0
        return loss_dict