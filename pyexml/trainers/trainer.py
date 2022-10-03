import torch
import torch.nn as nn
from ..geometry.maps.utils import optimal_diffeomorphism_LSA
from pyexlab.utils import get_info
from ..datasets.utils import DataMode
import numpy as np
import copy

class Trainer():
    __name__ = "Trainer"

    def __init__(self, model, dataset, criterion, optimizer, scheduler, alt_name=None, model_save_mod = 5):
        
        self.dataset = dataset
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=4, shuffle=True)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model = model
        self.model_save_mod = model_save_mod

        self.info_dict = {}
        self.info_dict['Model State'] = []

        if not alt_name is None:
            self.__name__ = alt_name
        
    def update(self, **kwargs):
        self.scheduler.step()

    def __call__(self, **kwargs):

        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=4, shuffle=True)

        total_loss = 0
        num_batches = len(self.dataloader)

        for batch_idx, samples in enumerate(self.dataloader):

            v = self.model(samples[0])   #v - Model output, u is expected output. Returned by model for better abstraction to isolate Trainer from model dependent sample handling.

            #if len(samples[:-1]) == 1:
            #    v = model(samples[0])
            #    u = samples[-1].float()
            #else:
            
            #u = torch.unsqueeze(samples[-1].float(), dim=1)

            loss = self.criterion(v, samples[1])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / num_batches

    def info(self, epoch = 0):

        self.info_dict['Name'] = self.__name__
        self.info_dict['Dataset Info'] = get_info(self.dataset)
        self.info_dict['Model Info'] = get_info(self.model)
        self.info_dict['Criterion Name'] = type(self.criterion).__name__
        self.info_dict['Optimizer Name'] = type(self.optimizer).__name__
        self.info_dict['Scheduler Name'] = type(self.scheduler).__name__
        
        if self.model_save_mod != -1 and epoch % self.model_save_mod == 0:
            self.info_dict['Model State'].append(copy.deepcopy(self.model.state_dict()))

        return self.info_dict

class Tester():

    __name__ = "Tester"

    def __init__(self, model, dataset, criterion, alt_name = None):
        
        self.dataset = dataset
        self.model = model
        self.criterion = criterion

        self.info_dict = {}
        
        if not alt_name is None:
            self.__name__ = alt_name

    def __call__(self, **kwargs):

        if 'vectorized' in kwargs and kwargs['vectorized'] == True:
            return self.vectorized_loss()
        else:
            return self.loss()

    def loss(self):
        
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=4, shuffle=True)

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
    
    def update(self, **kwargs):
        pass

    def info(self, epoch = 0):

        self.info_dict['Name'] = self.__name__
        self.info_dict['Dataset Info'] = get_info(self.dataset)
        self.info_dict['Model Info'] = get_info(self.model)
        self.info_dict['Criterion Name'] = type(self.criterion).__name__

        return self.info_dict

class DynamicLSATrainer(Trainer):

    def __init__(self, model, dataset, criterion, optimizer, scheduler, epoch_mod = -1, alt_name=None):

        
        if alt_name is None:
            alt_name = "DynamicLSATrainer"
        super().__init__(model, dataset, criterion, optimizer, scheduler, alt_name=alt_name)

        

        self.epoch_mod = epoch_mod

        self.info_dict['Epoch Mod'] = self.epoch_mod
        self.info_dict['Assignments'] = [self.dataset.get_assignment()]
    
    def update(self, epoch):

        super().update()

        if self.epoch_mod != -1 and epoch % self.epoch_mod == 0:

            new_assignment = optimal_diffeomorphism_LSA(self.dataset.domain, self.dataset.image, self.model)    #Maybe make universal class with function pointer for assignments?
            self.dataset.reassign(new_assignment)
            self.info_dict['Assignments'].append(new_assignment)
        
        


    