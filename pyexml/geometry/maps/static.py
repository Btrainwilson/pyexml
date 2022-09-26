from ._map import Map

class StaticMap(Map):

    def __init__(self, init_assignment):
        super().__init__(init_assignment)
        
    def reassign(self, new_assignment):
        pass
        

