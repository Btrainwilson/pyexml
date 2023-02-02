import torch

def MercerSPD(x):
    """
    Computes the Mercer kernel and constructs a semidefinite positive matrix.

    See https://arxiv.org/pdf/1711.06540.pdf

    Parameters:
        x (torch.Tensor): A tensor of size (batch, num_points)

    Returns:
        torch.Tensor : A positive semidefinite matrix of size (batch, num_points, num_points) constructed using the MercerKernel
    """
    #Computes all pairwise distances
    x = torch.cdist(x.unsqueeze(2), x.unsqueeze(2), p=2)

    #Computes mean of all distances
    if len(x.shape) == 2:
        sigma = torch.mean(x)
    else:
        sigma = torch.mean(x,dim=[1, 2])

    #Returns SPD using Mercer Kernel
    return torch.exp(-1/2 * x / sigma.unsqueeze(1).unsqueeze(2))

class MercerSPDModule(torch.nn.Module):
    """
    torch module that computes the Mercer kernel and constructs a semidefinite positive matrix.

    See https://arxiv.org/pdf/1711.06540.pdf

    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        forward function for pytorch module

        Parameters:
            x (torch.Tensor): A tensor of size (batch, num_points)

        Returns:
            torch.Tensor : A positive semidefinite matrix of size (batch, num_points, num_points) constructed using the MercerKernel
    
        """
        return MercerSPD(x)

if __name__ == "__main__":
    #Test Mercer Kernel
    a = torch.tensor([[[1, 2], [3, 4]],[[1, 3], [2, 4]]], dtype=torch.float32)
    l = MercerSPD(a)
    pass