import copy
from ...backend import device as backend_device
import torch
import numpy as np


class Map():
    """A map is a list of indices that can be used to map a subset of a dataset to a subset of another dataset."""

    def __init__(self, init_assignment, device=backend_device):
        self.reassign(init_assignment, device=device)


    def reassign(self, init_assignment, device=backend_device):

        if type(init_assignment) == int:
            self.assignment = torch.arange(0, init_assignment, device=device, dtype=torch.long)

        elif type(init_assignment) == list:
            self.assignment = torch.tensor(init_assignment, device=device, dtype=torch.long)
        
        elif type(init_assignment) == torch.Tensor:
            self.assignment = init_assignment

        elif type(init_assignment) == Map:
            self.assignment = copy.deepcopy(init_assignment.assignment)
        
        elif type(init_assignment) == np.ndarray:
            self.assignment = torch.from_numpy(init_assignment, device=device, dtype=torch.long)

        else:
            raise ValueError("Map must be initialized with a list, int, or torch.Tensor")

        if self.assignment.dtype != torch.long:
            self.assignment = self.assignment.long()
        
        if self.assignment.device != device:
            self.assignment = self.assignment.to(device)

        #Checks if sorted assignment is isomorphic to a range
        if not torch.equal(self.assignment, torch.arange(0, len(self.assignment), device=device, dtype=torch.long)):
            raise ValueError("Map must be initialized with a sorted list of indices that is isomorphic to a range, i.e., [0, 1, 2, 3, ...]")

        
    def __getitem__(self, idx):
        return self.assignment[idx]

