import pyexlab as pylab

class NeuralNetSubject(pylab.TestSubject):
    __name__ = "NeuralNetSubject"
    def __init__(self, trainer, alt_name = None):
        
        super().__init__(name=alt_name)

        self.trainer = trainer
        
        #Initialize Neural Net Values
        self.test_dict['time']['loss'] = []

        self.test_dict['info']['trainer_name'] = self.trainer.__name__
        self.test_dict['info']['trainer_info'] = self.trainer.info()

    def time_measure(self, epoch):

        #Measure state at current epoch
        super().time_measure(epoch)

        self.test_dict['time']['loss'].append(self.trainer(epoch = epoch))
        self.test_dict['time']['trainer_info'] = self.trainer.info()

        self.trainer.update(epoch)

        output_str = pylab.utils.dict_str_output(self.test_dict['time']['loss'][-1])
        
        return "%s Loss\n" %(self.__name__) + output_str
        

class CoachNetSubject(NeuralNetSubject):
    __name__ = "CoachNetSubject"