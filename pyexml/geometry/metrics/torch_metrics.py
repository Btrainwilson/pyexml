from .metric import TensorOperator
import torch

class Euclidean(TensorOperator):
    __name__ = "Euclidean"
    def across(self, X, Y):
        return torch.sum(torch.square(X - Y), dim=[1])

    def alltoall(self, X, Y):
        return self.across(X, Y)