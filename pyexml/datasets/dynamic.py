from torch.utils.data import Dataset
import pyexlab.fileio as fio

class DynamicDataset(Dataset):
    __name__ = "DynamicDataset"
    def __init__(self, dataset):
        self.info_dict = {}
        

        if type(dataset) == str:
           self.dataset = fio.load_dataset(dataset)

        self.dataset = dataset
        self.info_dict['len'] = len(self.dataset)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def update(self, **kwargs):
        pass

    def info(self):
        return self.info_dict

