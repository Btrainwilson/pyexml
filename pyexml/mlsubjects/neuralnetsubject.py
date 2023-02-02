import pyexlab as pylab
import numpy as np
import torch
import functools
from ..utilities import get_cuda
from ..trainers import Trainer, Tester
from ..datasets.utils import split_indeces
from ..datasets.dynamic import SimpleDataset
from ..datasets._maps import MapDataset
from ..backend import device as backend_device

class NeuralNetSubject(pylab.TestSubject):
    __name__ = "NeuralNetSubject"
    def __init__(self, trainers, alt_name = None):
        
        super().__init__(name=alt_name)

        self.trainers = trainers
        
        #Initialize Neural Net Values
        self.test_dict['Data'] = {}
        self.test_dict['Info']['Trainers'] = {}

        #Assign unique IDs to each trainer and query its information
        for idx, trainer in enumerate(self.trainers):
            trainer.id(idx)
            self.test_dict['Info']['Trainers'].update(trainer.info())

    def measure(self, epoch):

        #Measure state at current epoch
        super().measure(epoch=epoch)

        for trainer in self.trainers:
            self.test_dict['Data'].update(trainer(epoch = epoch))
            self.test_dict['Info']['Trainers'].update(trainer.info())
        
        out_str = ""
        for trainer_id in self.test_dict['Data']:
            out_str += self.test_dict['Data'][trainer_id]['Output String']

        return out_str

    def analysis(self):
        pass

class ModularNeuralNetSubject(NeuralNetSubject):

    def __init__(self, dataset_path, lr = 0.000001):

        self.dataset_path = dataset_path
        self.lr = lr

        self.init_dataset()
        self.init_models()
        self.init_trainers()

        super(ModularNeuralNetSubject, self).__init__(self.trainers)

    def init_dataset(self):
        pass

    def init_trainers(self):

        #Construct scheduler
        self.scheduler = functools.partial(torch.optim.lr_scheduler.ExponentialLR, gamma=0.99)

        #Construct ADAM optimizer for adaptive gradient descent
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        #Build trainer and tester
        self.model_trainer = Trainer(model = self.model, 
                                    dataset = self.training_dataset, 
                                    criterion = self.model.loss, 
                                    optimizer = self.optimizer, 
                                    scheduler = self.scheduler(self.optimizer), 
                                    state_save_mod=-1)

        self.model_tester  = Tester(model=self.model, 
                                    dataset = self.testing_dataset, 
                                    criterion = self.model.loss)
        
        #Construct the trainer list
        self.trainers = [self.model_trainer, self.model_tester]

    def init_models(self):
        pass

class SimpleNetSubject(ModularNeuralNetSubject):

    def __init__(self, data, labels, model, lr=0.0001, train_test_r = 0.8):

        self.model = model
        self.dataset = data
        self.labels = labels
        self.train_test_ratio = train_test_r

        super(SimpleNetSubject, self).__init__("", lr)

    def init_dataset(self):

        #Compute training set size
        training_size = int(len(self.dataset) * self.train_test_ratio)

        #Split Indeces for datasets
        subset_idx = split_indeces(len(self.dataset), training_size, len(self.dataset))

        #Create training and testing datasets 
        self.training_dataset = MapDataset(self.dataset[subset_idx[0]], self.labels[subset_idx[0]], device=backend_device)
        self.testing_dataset = MapDataset(self.dataset[subset_idx[1]], self.labels[subset_idx[1]], device=backend_device)
