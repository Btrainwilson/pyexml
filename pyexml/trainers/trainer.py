import torch
import torch.nn as nn
from ..geometry.maps.utils import optimal_diffeomorphism_LSA
from pyexlab.utils import get_info
from ..datasets.utils import DataMode
import numpy as np
import copy

class Trainer():

    __name__ = "Trainer"

    def __init__(self, model, dataset, criterion, optimizer, scheduler, alt_name=None, batch_size = 40):
        
        self.dataset = dataset
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model = model
        self.batch_size = batch_size

        self.info_dict = {}
        self.info_dict['Name'] = self.__name__
        self.info_dict['Dataset Info'] = get_info(self.dataset)
        self.info_dict['Model Info'] = get_info(self.model)
        self.info_dict['Criterion Name'] = type(self.criterion).__name__
        self.info_dict['Optimizer Name'] = type(self.optimizer).__name__
        self.info_dict['Scheduler Name'] = type(self.scheduler).__name__
        self.info_dict['Batch Size'] = self.batch_size

        self.call_dict = {}
        self.call_dict['Loss'] = []
        self.call_dict['Model State'] = []
        self.call_dict['Optimizer State'] = []

        if not alt_name is None:
            self.__name__ = alt_name

        
        
    def __call__(self, **kwargs):

        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        total_loss = 0
        num_batches = len(self.dataloader)

        for batch_idx, samples in enumerate(self.dataloader):

            v = self.model(samples[0])   #v - Model output, u is expected output. Returned by model for better abstraction to isolate Trainer from model dependent sample handling.
            loss = self.criterion(v, samples[1])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        

        self.call_dict['Loss'].append(total_loss / num_batches)
        self.call_dict['Model State'].append(copy.deepcopy(self.model.state_dict()))
        self.call_dict['Optimizer State'].append(copy.deepcopy(self.optimizer.state_dict()))

        self.scheduler.step()

        return self.call_dict 

    def info(self, epoch = 0):
        return self.info_dict

    def id(self, idx):
        return self.__name__ + str(idx)



class Tester():

    __name__ = "Tester"

    def __init__(self, model, dataset, criterion, alt_name = None, batch_size = 40):
        
        self.dataset = dataset
        self.model = model
        self.criterion = criterion
        self.batch_size = batch_size

        self.info_dict = {}
        self.info_dict['Name'] = self.__name__
        self.info_dict['Dataset Info'] = get_info(self.dataset)
        self.info_dict['Model Info'] = get_info(self.model)
        self.info_dict['Criterion Name'] = type(self.criterion).__name__

        self.call_dict = {}
        self.call_dict['Loss'] = []
        self.call_dict['Model State'] = []

        if not alt_name is None:
            self.__name__ = alt_name

    def __call__(self, **kwargs):

        if 'vectorized' in kwargs and kwargs['vectorized'] == True:
            self.call_dict['Loss'].append(self.vectorized_loss())
            
        else:
            self.call_dict['Loss'].append(self.loss())

        self.call_dict['Model State'].append(copy.deepcopy(self.model.state_dict()))
        return self.call_dict

    def loss(self):
        
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        loss_sum = 0
        num_batches = len(self.dataloader)

        with torch.no_grad():
            for batch_idx, samples in enumerate(self.dataloader):

                v = self.model(samples[0])
                loss_sum += self.criterion(v, samples[1]).item()

        loss_sum /= num_batches

        return loss_sum

    def vectorized_loss(self):

        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=1, shuffle=False)
        vec = np.zeros(len(self.dataloader))

        with torch.no_grad():
            for batch_idx, samples in enumerate(self.dataloader):
                v = self.model(samples[0])
                vec[batch_idx] = self.criterion(v, samples[1]).item()

        return vec

    def info(self, epoch = 0):
        return self.info_dict

    def id(self, idx):
        return self.__name__ + str(idx)

class DynamicLSATrainer(Trainer):
    __name__ = "DynamicLSATrainer"
    def __init__(self, model, dataset, criterion, optimizer, scheduler, epoch_mod = -1, alt_name=None):

        if alt_name is None:
            alt_name = "DynamicLSATrainer"

        super().__init__(model, dataset, criterion, optimizer, scheduler, alt_name=alt_name)

        self.epoch_mod = epoch_mod

        self.info_dict['Epoch Mod'] = epoch_mod
        self.call_dict['Assignments'] = [self.dataset.get_assignment()]

    def __call__(self, **kwargs):

        super().__call__(**kwargs)

        if (self.epoch_mod != -1) and ('epoch' in kwargs) and (kwargs['epoch'] % self.epoch_mod == 0):

            new_assignment = optimal_diffeomorphism_LSA(self.dataset.domain, self.dataset.image, self.model)    #Maybe make universal class with function pointer for assignments?
            self.dataset.reassign(new_assignment)
            self.call_dict['Assignments'].append(new_assignment)

        return self.call_dict


        
        


    