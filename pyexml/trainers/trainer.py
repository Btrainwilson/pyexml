import torch
import torch.nn as nn
from ..geometry.maps.utils import optimal_diffeomorphism_LSA
from pyexlab.utils import get_info
from ..datasets.utils import DataMode
import numpy as np
import copy
import torch.nn.functional as F

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

###Michael Bezick's Code###

#1. How do I handle Cuda?
#2. Does the dataloader need to tensor transform and normalization transform?

class cGANTrainer(Trainer):
    __name__ = "cGANTrainer"
    def __init__(self, model, dataset, criterion, optimizer, scheduler, n_classes, z_dim, epoch_mod = -1, alt_name=None, batch_size = 32, device = 'cpu'): 
             
        if alt_name is None:
            alt_name = "cGANTrainer"

        super().__init__(model, dataset, criterion, optimizer, scheduler, alt_name=alt_name, batch_size = batch_size)

        self.Generator = model.Generator
        self.Discriminator = model.Discriminator
        self.device = device
        self.n_classes = n_classes
        self.z_dim = z_dim
        self.Generator_Optimizer = optimizer.Generator_Optimizer
        self.Discriminator_Optimizer = optimizer.Discriminator_Optimizer

    def __call__(self, **kwargs):

        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        
        total_gen_loss = 0
        total_disc_loss = 0

        num_batches = len(self.dataloader)

        ##my code##
        for real, labels in self.dataloader:
            
            #creating and concatenating one hot labels
            one_hot_labels = get_one_hot_labels(labels.to(self.device), self.n_classes)
            image_one_hot_labels = one_hot_labels[:,:, None, None]
            image_one_hot_labels = image_one_hot_labels.repeat(1, 1, self.dataset_shape[1], self.dataset_shape[2])
            
            self.Discriminator_Optimizer.zero_grad()

            fake_noise = get_noise(self.batch_size, self.z_dim, device=self.device)
            noise_and_labels = combine_vectors(fake_noise, one_hot_labels)

            #generation of fake
            fake = self.Generator(noise_and_labels)

            #combining images and labels for discriminator
            fake_image_and_labels = combine_vectors(fake, image_one_hot_labels)
            real_image_and_labels = combine_vectors(real, image_one_hot_labels)
           
            #using discriminator
            disc_fake_pred = self.Discriminator(fake_image_and_labels.detach())
            
            disc_real_pred = self.Discriminator(real_image_and_labels)
            
            #calculating loss
            #zeros and ones vector tells discriminator whether images are fake or real
            disc_fake_loss = self.criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
            disc_real_loss = self.criterion(disc_real_pred, torch.ones_like(disc_real_pred))

            #averaging both losses
            disc_loss = (disc_fake_loss + disc_real_loss) / 2

            #backpropagation through discriminator
            disc_loss.backward(retain_graph=True)
            self.Discriminator_Optimizer.step()
            
            ###Updating Generator###

            #zeroing out gradient
            self.Generator_Optimizer.zero_grad()
    
            #combining fake images with broadcast labels
            fake_image_and_labels = combine_vectors(fake, image_one_hot_labels)

            #calculating loss for fakes
            disc_fake_pred = self.discriminator(fake_image_and_labels, im_chan = self.disc_im_chan, 
                                                hidden_dimensions = self.disc_hidden_dimensions, 
                                                kernel_size = self._disc_kernel_size, stride = self.disc_stride)
            
            gen_loss = self.criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))

            #backpropagation through generator
            gen_loss.backward()
            self.Generator_Optimizer.step()

        self.call_dict['Generator Loss'].append(gen_loss / num_batches)
        self.call_dict['Discriminator Loss'].append(disc_loss / num_batches)
        self.call_dict['Generator State'].append(copy.deepcopy(self.generator.state_dict()))
        self.call_dict['Discriminator State'].append(copy.deepcopy(self.discriminator.state_dict()))
        self.call_dict['Optimizer State'].append(copy.deepcopy(self.optimizer.state_dict()))

        self.scheduler.step()

        return self.call_dict

def get_noise(n_samples, input_dim, device='cpu'):
    return torch.randn(n_samples, input_dim, device=device)

def get_one_hot_labels(labels, n_classes):
    return F.one_hot(labels, n_classes)

def combine_vectors(x, y):
    combined = torch.cat((x.float(), y.float()), 1)
    return combined

def get_input_dimensions(z_dim, dataset_shape, n_classes):
  '''
  z_dim: the length of the noise vector
  dataset_shape: the shape of the dataset images (Channels, Width, Height)
  n_classes: the number of classes in dataset
  '''
  generator_input_dim = z_dim + n_classes
  discriminator_im_chan = dataset_shape[0] + n_classes
  im_chan = dataset_shape[0]
  return generator_input_dim, discriminator_im_chan, im_chan
