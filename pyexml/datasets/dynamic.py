from torch.utils.data import Dataset
import pyexlab.fileio as fio

class SimpleDataset(Dataset):
    __name__ = "SimpleDataset"
    def __init__(self, dataset):
        
        if type(dataset) == str:
           self.dataset = fio.load_dataset(dataset)

        self.dataset = dataset

        self.info_dict = {}
        self.info_dict['Name'] = self.__name__
        self.info_dict['Length'] = len(self.dataset)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def update(self, **kwargs):
        pass

    def info(self):
        return self.info_dict


