import torch
import torch.distributions as dist
from .energyfunctions import partition_fn, boltzmann_factor, qubo_energy
from ..utilities.data import int_to_bit_strings
import functools

class EnergyExact(dist.Distribution):
    """
    Class representing an exact, energy based distribution. This class precomputes the energy of all configurations.
    
    Parameters:
        energy_fn (function) : function that takes in a tensor of configurations and returns the corresponding energy.
        num_spins (int) : int representing the number of spins in the system.
        validate_args (bool): whether to validate input with asserts
    """
    def __init__(self, energy_fn, num_spins, validate_args=None):
        """
        Initialize the distribution.
        """
        #System size
        self.num_spins = num_spins

        #Specify the energy function
        self.energy_fn = energy_fn

        #Compute the partition function and energies of all configurations
        self.partition, self.energies = partition_fn(energy_fn, num_spins)

        #Compute probability of all configurations
        self.prob = self.energies / self.partition

        #Compute entropy
        self.entropy = torch.sum(self.prob * torch.log(self.prob))

        super(EnergyExact, self).__init__(validate_args=validate_args)

    def sample(self, sample_shape=torch.tensor([1])):
        """
        Draw a sample from the distribution.
        
        Parameters:
            sample_shape (torch.Size): shape of the sample.
        
        Returns:
            torch.Tensor : a sample from the distribution.
        """
        
        # Sample from the multinomial distribution
        if type(sample_shape) is int:
            sample_shape = torch.tensor([sample_shape])

        samples = torch.multinomial(self.prob, num_samples=int(torch.prod(sample_shape)))
        
        # Convert the samples to bit string vectors
        bit_strings = torch.Tensor(int_to_bit_strings(samples, self.num_spins))

        # Reshape the samples
        if len(sample_shape) == 1:
            return bit_strings
        else:
            sample_shape.append(self.num_spins)
            reshaped_samples = bit_strings.view(sample_shape)
            return reshaped_samples

    def log_prob(self, x):
        """
        Compute the log-probability of a set of configurations under the energy distribution.
        
        Parameters:
            x (torch.Tensor) : a tensor of configurations.
        
        Returns:
            torch.Tensor : log-probability of the configurations.
        """
        return torch.log(self.energy_fn(x) / self.partition)

    def entropy(self):
        """
        Computes the entropy of the distribution.
        """
        return self.entropy

class BoltzmannExact(EnergyExact):
    """
    Class representing the Boltzmann distribution of a QUBO system.
    
    Parameters:
        H (torch.Tensor(num_spins, num_spins)): Tensor representing the QUBO matrix of the system.
        num_spins (int): Number of spins in the system.
        temperature (float): Temperature of the system.
        validate_args (bool, optional): Whether to validate input with asserts. Default: None.
    """
    def __init__(self, H, num_spins, temperature, validate_args=None):
        """
        Initialize the distribution.
        """
        self.H = H
        self.temperature = temperature
        energy_fn = functools.partial(boltzmann_factor, H = H, temperature=temperature)

        super(BoltzmannExact, self).__init__(energy_fn, num_spins, validate_args=validate_args)

    def log_prob(self, x):
        """
        Computes the log probability of a given configuration x.
        """
        return -qubo_energy(x, self.H) / self.temperature - torch.log(self.partition)





