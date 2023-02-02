import torch

def get_cuda():
    """
    Returns the cuda device if it is available. Else returns the CPU.
    """
    if torch.cuda.is_available(): 
        dev = "cuda:0" 
    else: 
        dev = "cpu" 

    return torch.device(dev) 

def parallelize_model(model, device_ids):
    """
    Wrapper function to parallelize a PyTorch model on a GPU cluster.
    :param model: the model to parallelize
    :param device_ids: a list of GPU IDs to use for parallelization
    :return: the parallelized model
    """
    model = model.to(device_ids[0])
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    return model

def get_best_device():
    
    cuda_capability = torch.cuda.get_device_capability(torch.cuda.current_device())

def choose_best_device():
    """
    Function that checks the PyTorch version, Tensor Core, and GPU functionality and chooses the best available device.
    It also enables the Tensor Core flags if tensor cores are available
    :return: a device (cpu or cuda)
    """

    # Check if CUDA is available
    if torch.cuda.is_available():
        # Get the number of GPUs available
        n_gpus = torch.cuda.device_count()

        # Get the maximum GPU compute capability
        max_compute_capability = max(torch.cuda.get_device_capability(i) for i in range(n_gpus))
        
        # Check Pytorch version
        version = torch.__version__.split(".")
        if int(version[0]) >= 1 and int(version[1]) >= 4:

            # Tensor Cores are supported in this Pytorch version
            if max_compute_capability[0] >= 7:
                # Tensor Cores are supported in this GPU
                # Enable Tensor Core support
                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.benchmark = True
                
            return torch.device("cuda")

    else:
        return torch.device("cpu")

# Example usage:
device = choose_best_device()
print(f"The best available device is {device}")


# Example usage:
device = choose_best_device()
print(f"The best available device is {device}")
