from .neuralnetsubject import NeuralNetSubject

class DynamicNetSubject(NeuralNetSubject):
    __name__ = "DynamicNetSubject"
    def __init__(self, model, trainer):
        super().__init__(model, trainer)

    def time_measure(self, epoch):
        super().time_measure(epoch)
        