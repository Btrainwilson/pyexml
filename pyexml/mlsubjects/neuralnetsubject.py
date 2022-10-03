import pyexlab as pylab

class NeuralNetSubject(pylab.TestSubject):
    __name__ = "NeuralNetSubject"
    def __init__(self, trainer, alt_name = None):
        
        super().__init__(name=alt_name)

        self.trainer = trainer
        
        #Initialize Neural Net Values
        self.test_dict['Measurements']['Loss'] = []

        self.test_dict['Info']['Trainer'] = self.trainer.info()

    def measure(self, epoch):

        #Measure state at current epoch
        super().measure(epoch=epoch)

        self.test_dict['Measurements']['Loss'].append(self.trainer(epoch = epoch))
        self.test_dict['Info']['Trainer'] = self.trainer.info()

        self.test_dict['Measurements']['Trainer Measurement'] = self.trainer.update(epoch)

        output_str = pylab.utils.dict_str_output(self.test_dict['Measurements']['Loss'][-1])
        
        return "%s Loss\n" %(self.__name__) + output_str
        
