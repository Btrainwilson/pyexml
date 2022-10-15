from torch.utils.data import Dataset
import torch
from ..geometry import maps
from .utils import split_indeces, DataMode


class MapDataset(Dataset):

    def __init__(self, domain, image, device=None, assignment=None, subspace = None):

        self.info_dict = {}
        self.info_dict['Name'] = "MapDataset"
        self.info_dict['Assignments'] = []
        self.info_dict['Subspace'] = subspace

        if subspace is not None:
            self.domain = domain[subspace]
            self.image = image[subspace]
        else:
            self.domain = domain
            self.image = image

        self.domain = torch.tensor(self.domain, dtype=torch.float32)
        self.image = torch.tensor(self.image, dtype=torch.float32)

        self.subspace = subspace

        if assignment is None:
            self.assignment = maps.Map(list(range(len(domain))))
            self.info_dict['Assignments'] = [list(range(len(domain)))]
        else:
            self.assignment = assignment
            self.info_dict['Assignments'] = [assignment]

        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device
        
        self.info_dict['device'] = self.device

    def __getitem__(self, idx):

        idx_hat = self.assignment[idx]
        return [self.domain[idx], self.image[idx_hat]]


    def reassign(self, new_assign):
        self.info_dict['Assignments'].append(new_assign)
        self.assignment.reassign(new_assign)

    def __len__(self):
        return len(self.domain)

    def update(self, **kwargs):
        pass

    def info(self):
        return self.info_dict

    def get_assignment(self):
        return self.assignment.assignment
    
    def setmode(self, mode):
        self.mode = mode