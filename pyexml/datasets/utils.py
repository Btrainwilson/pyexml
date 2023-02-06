import random 
import enum
import os
import torchvision
import numpy as np
import torch


from pyexlab import fileio as fio
from ..backend import device as backend_device
from ..backend import default_dtype as backend_dtype

class DataMode(enum.Enum):
    TRAIN = 1
    TEST = 2
    FULL = 3

def split_indeces(dataset_size, split_size=0.8, subspace_size=1.0):

    if type(subspace_size) == int and subspace_size > dataset_size or type(subspace_size) == float and (subspace_size > 1.0 or subspace_size < 0.0):
        raise ValueError("subspace_size must be an integer less than dataset_size or float between 0 and 1")

    if type(split_size) == int and split_size > dataset_size or type(split_size) == float and (split_size > 1.0 or split_size < 0.0):
        raise ValueError("split_size must be an integer less than dataset_size or float between 0 and 1")

    if type(subspace_size) == float and subspace_size <= 1.0 and subspace_size > 0.0:
        subspace_size = int(dataset_size * subspace_size)

    if type(split_size) == float and split_size <= 1.0 and split_size > 0.0:
        split_size = int(dataset_size * split_size)

    if split_size > subspace_size:
        raise ValueError("split_size must be less than or equal to subspace_size")

    rearranged_index = list(range(dataset_size))
    random.shuffle(rearranged_index)
    return [rearranged_index[:split_size], rearranged_index[split_size:subspace_size]]

def split_dataset(dataset, split_size, subspace_size=1.0):

    split_idx = split_indeces(len(dataset), split_size, subspace_size)

    return dataset[split_idx[0]], dataset[split_idx[1]]

def load_dataset_to_pymltorch(dataset, device=backend_device, dtype=backend_dtype, **kwargs):
    """Converts a dataset to a pyml dataset
    Args:
        dataset (str, np.ndarray, torch.Tensor, torchvision.datasets, dict, list): The dataset to convert. Can be dictionary or list of other valid dataset types
        device (torch.device): The device to convert the dataset to
        dtype (torch.dtype): The dtype to convert the dataset to
        **kwargs: Additional arguments to pass to torchvision.datasets
    Returns:
        dict: A dictionary containing the dataset and targets (if they exist)

    """

    dataset_dict = {}

    if type(dataset) == str and os.path.exists(dataset):
        dataset = fio.load_dataset(dataset)

    elif type(dataset) == str and is_torchvision_dataset(dataset):

        dataset = get_torchvision_dataset(dataset, **kwargs)

        dataset_dict['data'] = load_to_device(dataset.data, device=device, dtype=dtype)
        dataset_dict['targets'] = load_to_device(dataset.targets, device=device, dtype=dtype)

        #Check if dataset targets are one hot encoded
        if not (len(dataset_dict['targets'].shape) == 2 and dataset_dict['targets'].shape[1] == 1):
            dataset_dict['targets'] = torch.functional.F.one_hot(dataset_dict['targets'].long())

        #Fix channel dimension if necessary
        dataset_dict['data'] = fix_channels(dataset_dict['data'])


    elif type(dataset) == np.ndarray or type(dataset) == torch.Tensor or type(dataset) == list:
        dataset_dict['data'] = load_to_device(dataset, device=device, dtype=dtype)

    elif type(dataset) == dict:
        for key, value in dataset.items():
            dataset_dict[key] = load_dataset_to_pymltorch(value, device=device, dtype=dtype, **kwargs)

    else:
        raise ValueError("Dataset type loading not supported (must be str, np.ndarray, torch.Tensor, torchvision.datasets, dict, list)")

    return dataset_dict


def load_to_device(dataset, device=backend_device, dtype=backend_dtype):
    """Loads dataset to device
    Args:
        dataset (np.ndarray, torch.Tensor): The dataset to convert
        device (torch.device): The device to convert the dataset to
        dtype (torch.dtype): The dtype to convert the dataset to
    Returns:
        torch.Tensor: The dataset loaded to the device
    """

    if type(dataset) == np.ndarray:
        dataset = torch.from_numpy(dataset)
        dataset = dataset.to(device=device, dtype=dtype)

    elif type(dataset) == torch.Tensor:
        dataset = dataset.to(device=device, dtype=dtype)

    elif type(dataset) == list:
        dataset = torch.tensor(dataset, device=device, dtype=dtype)

    else:
        raise ValueError("Dataset type loading not supported (must be np.ndarray, torch.Tensor)")

    return dataset

def is_torchvision_dataset(dataset_name):
    return dataset_name in torchvision.datasets.__all__

def get_torchvision_dataset(dataset_name, **kwargs):
    return getattr(torchvision.datasets, dataset_name)(**kwargs)

def fix_channels(dataset):
    if len(dataset.shape) == 3:
        dataset = dataset.unsqueeze(1)
    elif len(dataset.shape) == 4 and dataset.shape[3] == 3:
        dataset = dataset.permute(0, 3, 1, 2)
    return dataset


#Test code
if __name__ == "__main__":
    #Load datasets 
    train_dataset = load_dataset_to_pymltorch('mnist', train=True, download=True)
    test_dataset = load_dataset_to_pymltorch('mnist', train=False, download=True)
