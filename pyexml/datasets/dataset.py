from torch.utils.data import Dataset
import pyexlab.fileio as fio
import numpy as np
import torch

from ..backend import device as backend_device
from ..backend import default_dtype as backend_dtype
from ..backend.pyml_object import PYMLObject
from ..geometry.maps import Map

from .utils import load_dataset_to_pymltorch

class Dataset(Dataset, PYMLObject):
    """Dataset class for PyML
    Args:
        data (str, np.ndarray, torch.Tensor, torchvision.datasets, dict, list): The dataset to convert. Can be dictionary or list of other valid dataset types
        target (str, np.ndarray, torch.Tensor, torchvision.datasets, dict, list): The target to convert. Can be dictionary or list of other valid dataset types
        device (torch.device): The device to convert the dataset to
        dtype (torch.dtype): The dtype to convert the dataset to
        map_assignment (pyexml.geometry.maps.Map, list, int): The map between the dataset and the target, 
    """

    def __init__(self, data, target=None, device = backend_device, dtype = backend_dtype, map_assignment=None, map_subspace=None, name="Dataset", **kwargs):

        # Initialize PYMLObject
        PYMLObject.__init__(self, name=name, device=device, dtype=dtype, **kwargs)

        # Set map_subspace
        self.map_subspace = map_subspace
        self.info_dict['Subspace'] = self.map_subspace

        #Load data and target (if given) to tensors with correct device and dtype
        loaded_dataset = load_dataset_to_pymltorch(data, device=self.device, dtype=self.dtype, **kwargs)
        self.data = loaded_dataset['data']

        # If map_subspace is given, apply it to the data
        if self.map_subspace is not None:
            self.data = self.data[self.map_subspace]

        if 'targets' in loaded_dataset:
            self.target = loaded_dataset['targets']

        else:
            # If target is given, load it to tensor with correct device, dtype, and apply map_subspace
            if target is not None:
                loaded_target = load_dataset_to_pymltorch(target, device=device, dtype=dtype, **kwargs)
                self.target = loaded_target['data']

                if self.map_subspace is not None:
                    self.target = self.target[self.map_subspace]
            else:
                self.target = None

        # Set map_assignment
        self.map_assignment = map_assignment

        # Create map_assignment if not given
        if self.map_assignment is None:
            self.map_assignment = Map(len(self.data), device=self.device)
        else:
            self.map_assignment = Map(self.map_assignment, device=self.device)
        
        self.info_dict['Assignments'] = [self.map_assignment.assignment]

    def __getitem__(self, idx):
        """Returns the data and target at the given index"""

        if self.target is not None:
            idx_hat = self.map_assignment[idx]
            return [self.data[idx], self.target[idx_hat]]
        else:
            return [self.data[idx]]

    def reassign(self, new_assign):
        """Reassigns the map_assignment to the given assignment"""
        
        self.map_assignment.reassign(new_assign)
        self.info_dict['Assignments'].append(self.map_assignment.assignment)

    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.data)

    def update(self, **kwargs):
        """Updates the dataset with the given keyword arguments"""
        pass

    def split(self, split_size, shuffle=True, seed=None):
        """Splits the dataset into two datasets of the given split_size"""
        
        if seed is not None:
            np.random.seed(seed)

        #Check if split_size is an integer
        if split_size < 1 or split_size > 0:
            #If split_size is a float, convert it to an integer
            split_size = int(split_size * len(self))

        #Check if split_size is valid
        if split_size > len(self):
            raise ValueError("Split size is larger than dataset size")

        if split_size < 0:
            raise ValueError("Split size is negative")

        #Create a list of indices
        indices = torch.arange(len(self), device=self.device)

        #Shuffle the indices if shuffle is True
        if shuffle:
            indices = torch.randperm(len(self), device=self.device)

        #Create the new map_assignment

        dataset1 = Dataset(self.data[indices[:split_size]], target=self.target[self.map_assignment[indices[:split_size]]], device=self.device, dtype=self.dtype, name=self.name + "_1")
        dataset2 = Dataset(self.data[indices[split_size:]], target=self.target[self.map_assignment[indices[split_size:]]], device=self.device, dtype=self.dtype, name=self.name + "_2")


        return dataset1, dataset2





        





