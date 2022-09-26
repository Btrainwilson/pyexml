from .trainer import Trainer

#Dictates the use of trainers
class Coach(Trainer):
    __name__ = "Coach_Trainer"
    def __init__(self, trainers, trainer_schedule):

        self.trainers = trainers
        self.trainer_schedule = trainer_schedule
        self.info_dict = {}
        
        
    def update(self, epoch):
        for t_i in self.trainer_schedule(epoch):
            self.trainers[t_i].update(epoch = epoch)

    def __call__(self, **kwargs):

        if 'epoch' in kwargs.keys():
            epoch = kwargs['epoch']
        else:
            epoch = 0

        train_loss = {}
        for t_i in self.trainer_schedule(epoch):
            trainer_name = str(t_i) + "_" + self.trainers[t_i].__name__ 
            train_loss[trainer_name] = self.trainers[t_i](epoch = epoch)

        return train_loss

    def info(self):
        
        self.info_dict['Name'] = 'CoachTrainer'
        self.info_dict['Trainers Info'] = {}

        for t_i, trainer in enumerate(self.trainers):
            self.info_dict['Trainers Info'][t_i] = trainer.info()
            
        return self.info_dict




