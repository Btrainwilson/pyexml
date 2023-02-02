import torch 

def neg_bce_loss(output, target, reduction='sum'):
    return -torch.functional.F.binary_cross_entropy(output, target, reduction=reduction)