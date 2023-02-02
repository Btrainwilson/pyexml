import torch

def SmoothRho(image, cbeta, chi):
    """
    Computes the smoothed value of the image using PyTorch tensor operations and broadcasting.

    Parameters:
    image (np.ndarray or torch.Tensor): A 2D array of the initial image values.
    cbeta (float): A constant used in the smoothing calculation.
    chi (np.ndarray or torch.Tensor): A 1D array of constants used in the smoothing calculation.

    Returns:
    torch.Tensor: A 2D tensor of smoothed image values.

    """
    image = torch.tensor(image, dtype=torch.float32)
    chi = torch.tensor(chi, dtype=torch.float32)
    cbeta = torch.tensor(cbeta, dtype=torch.float32)
    
    # Compute smoothed Rho values using broadcasting and tensor operations
    Rho = chi[None, None, :] * (torch.exp(-cbeta * (1 - image[..., None] / chi[None, None, :])) - (1 - image[..., None] / chi[None, None, :]) * torch.exp(-cbeta))
    Rho[image[..., None] > chi[None, None, :]] += (1 - chi[None, None, :]) * (1 - torch.exp(-cbeta * (image[..., None] - chi[None, None, :]) / (1 - chi[None, None, :]))) + (image[..., None] - chi[None, None, :]) / (1 - chi[None, None, :]) * torch.exp(-cbeta)

    # Clamp values to be between 0 and 1
    Rho = torch.clamp(Rho, 0, 1)
    
    # Remove the unnecessary third dimension
    return Rho.squeeze(-1)

