from .neuralnetsubject import NeuralNetSubject
from ..models.metricnet import MetricNet

class MetricNetSubject(NeuralNetSubject):

    def __init__(self, base_model, metric, trainer):
        super().__init__(MetricNet(base_model=base_model, metric=metric), trainer)

    def time_measure(self, epoch):
        super().time_measure(epoch)


    


        

        

    
