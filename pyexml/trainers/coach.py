from .trainer import Trainer

#Dictates the use of trainers. 
class Coach(Trainer):

    __name__ = "Coach_Trainer"
    def __init__(self, trainers, trainer_schedule):

        self.trainers = trainers
        self.trainer_schedule = trainer_schedule

        self.info_dict = {}
        self.info_dict['Name'] = 'CoachTrainer'
        self.info_dict['Team Info'] = {}

        self.call_dict = {}

        for t_i, trainer in enumerate(self.trainers):
            self.info_dict['Team Info'][trainer.id(t_i)] = trainer.info()
            self.call_dict[self.trainers[t_i].id(t_i)] = {}

        

    def __call__(self, **kwargs):

        if 'epoch' in kwargs.keys():
            epoch = kwargs['epoch']
        else:
            epoch = 0

        for t_i in self.trainer_schedule(epoch):
            self.call_dict[self.trainers[t_i].id(t_i)].update(self.trainers[t_i](epoch = epoch))

        return self.call_dict

    def info(self):
        return self.info_dict





