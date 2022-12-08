import pyexlab as pylab

class NeuralNetSubject(pylab.TestSubject):
    __name__ = "NeuralNetSubject"
    def __init__(self, trainer, alt_name = None):
        
        super().__init__(name=alt_name)

        self.trainer = trainer
        
        #Initialize Neural Net Values
        self.test_dict['Data'] = {}
        self.test_dict['Info']['Trainer'] = self.trainer.info()

    def measure(self, epoch):

        #Measure state at current epoch
        super().measure(epoch=epoch)

        self.test_dict['Data'].update(self.trainer(epoch = epoch))
        self.test_dict['Info']['Trainer'].update(self.trainer.info())
        
        out_str = ""
        for trainer_id in self.test_dict['Data']:
            out_str += "%s Loss: %f \n" % (trainer_id, self.test_dict['Data'][trainer_id]['Loss'][-1])

        return out_str

    def analysis(self):
        pass
