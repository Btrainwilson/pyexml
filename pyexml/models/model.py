#Wrapper class around torch.nn 
#May deprecate if torch.nn is sufficient. Standby.
import torch

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.__name__ = str(type(self))

        self.info_dict = {}
        self.info_dict['Name'] = self.__name__
        self.info_dict['state_dict'] = self.state_dict()
    
    def info(self):
        self.info_dict.update(self.__dict__)
        return self.info_dict