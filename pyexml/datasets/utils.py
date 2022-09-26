import random 
import enum

class DataMode(enum.Enum):
    TRAIN = 1
    TEST = 2
    FULL = 3
    
def split_indeces(length, split_index, total_size):
    rearranged_index = list(range(length))
    random.shuffle(rearranged_index)
    return [rearranged_index[:split_index], rearranged_index[split_index:total_size]]