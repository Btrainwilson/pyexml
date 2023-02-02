import torch
import itertools
import functools

def qubo_energy(x, H):
    """
    Computes the energy for the specified Quadratic Unconstrained Binary Optimization (QUBO) system.
    
    Parameters:
        x (torch.Tensor) : Tensor of shape (batch_size, num_dim) representing the configuration of the system.
        H (torch.Tensor) : Tensor of shape (batch_size, num_dim, num_dim) representing the QUBO matrix.
    
    Returns:
        torch.Tensor : The energy for each configuration in the batch.
    """
    if len(x.shape) == 1 and len(H.shape) == 2:
        return torch.einsum("i,ij,j->", x, H, x)
    elif len(x.shape) == 2 and len(H.shape) == 3:
        return torch.einsum("bi,bij,bj->b", x, H, x)
    elif len(x.shape) == 2 and len(H.shape) == 2:
        return torch.einsum("bi,ij,bj->b", x , H, x)
    else:
        raise ValueError("Invalid shapes for x and H. x must be of shape (batch_size, num_dim) and H must be of shape (batch_size, num_dim, num_dim).")


def boltzmann_factor(x, H, temperature = 1):
    """
    Computes the boltzmann factor for the specified system.
    
    Parameters:
        x (torch.Tensor) : Batched configuration tensor of shape (batch_size, num_dim)
        H (torch.Tensor) : Batched energy matrices of shape (batch_size, num_dim, num_dim)
        temperature (float) : Scalar representing the temperature of the system.
    
    Returns:
        torch.Tensor : The boltzmann factor for each configuration in the batch.
    """
    energies = qubo_energy(x, H)
    return torch.exp(-energies / temperature)



def rydberg_energy(positions, strengths, x, c6 = 1):
    """
    Computes the energy of a Rydberg system.

    Parameters:
        positions (torch.Tensor) : Tensor of shape (batch_size, num_atoms, 3) representing the positions of the atoms.
        strengths (torch.Tensor) : Tensor of shape (batch_size, num_atoms, num_atoms) representing the strengths of the interactions.
        x (torch.Tensor) : Tensor of shape (batch_size, num_atoms) representing if a site is in the Rydberg state.
        c6 (float) : Scalar representing the coefficient of the sixth power term in the interaction strength.
    
    Returns:
        torch.Tensor: Energies of all configurations in x.
    """

    differences = positions.unsqueeze(-2) - positions.unsqueeze(-3)
    distances = torch.norm(differences, dim=-1)
    x = x.unsqueeze(-1)
    energy = c6 * (strengths * (x.unsqueeze(-2) & x.unsqueeze(-3))) / distances**6
    return energy



def boltzmann_partition(num_spins, interactions, temperature):
    """
    Computes the exact Boltzmann partition function for the specified system.
    
    Parameters:
        num_spins (int) : int representing the number of spins in the system.
        interactions (torch.Tensor) : Tensor of shape (num_spins, num_spins) representing the interaction graph between spins.
        temperature (float) : Scalar representing the temperature of the system.
    
    Returns:
        float: Partition function.
        torch.Tensor : Energy vector of size (2**num_spins) that holds the energy of all configurations of the system.
    """
    energy_fn = functools.partial(boltzmann_factor, H = interactions, temperature=temperature)
    return partition_fn(energy_fn, num_spins)


def partition_fn(energy_fn, num_spins, basis=[0, 1]):
    """
    Computes the partition function for the specified system given an energy function.
    
    Parameters:
        energy_fn (function) : function that takes in a tensor of configurations and returns the corresponding energy.
        num_spins (int) : int representing the number of spins in the system.
        basis (list): list of possible states for each spin
    
    Returns:
        float : Partition function.
        torch.Tensor : Energy vector of size (2**num_spins) that holds the energy of all configurations of the system.
    """
    # Generate all configurations
    configurations = torch.FloatTensor(list(itertools.product(basis, repeat=num_spins)))

    # Query energy function on all configurations
    energies = energy_fn(configurations)

    # Compute partition function
    partition = energies.sum()

    return partition, energies



if __name__ == "__main__":
    x = torch.randint(2,4)
    H = torch.rand()
    