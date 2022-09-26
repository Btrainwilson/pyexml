import copy

class Map():

    def __init__(self, init_assignment):
        self.assignment = copy.deepcopy(init_assignment)

    def reassign(self, new_assignment):
        self.assignment = copy.deepcopy(new_assignment)
        
    def __getitem__(self, idx):
        return self.assignment[idx]

