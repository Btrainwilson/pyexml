import torch
from ..utilities.cuda import get_best_device

#Initialize package variables

#Default datatype for all models
default_dtype = torch.float32

#Default device for all models
device = get_best_device()

#Parallelize models onto cluster
cluster_enabled = False

#Print debug information
debug_console = False

#Print test information
test_console = False

