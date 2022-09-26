
class Trainer_Schedule():

    def __init__(self):
        pass

    def __call__(self, epoch):
        pass

class Modulus_Schedule(Trainer_Schedule):

    def __init__(self, mod_schedule):
        
        self.mod_schedule = mod_schedule

    def __call__(self, epoch):

        return [i for i, val in enumerate(self.mod_schedule) if val == -1 or epoch % val == 0]


