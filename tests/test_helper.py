import torch

def torch_equal(tensor1, tensor2):
    """Compares two tensors for equality"""
    #Convert tensor to bool
    data_result = (tensor1 == tensor2).cpu().numpy().all().astype(bool)
    dtype_result = tensor1.dtype == tensor2.dtype
    device_result = tensor1.device == tensor2.device


    return data_result and dtype_result and device_result

#Test helper functions
if __name__ == "__main__":

    print(torch_equal(torch.tensor([1,2,3]), torch.tensor([1,2,3])))
    print(torch_equal(torch.tensor(1), torch.tensor([0, 1])))
    print(torch_equal(torch.tensor(1, device="cuda"), torch.tensor(1)))