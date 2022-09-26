import scipy
import numpy as np
import torch

from ... import generators

class TensorOperator():

    def __call__(self, X, Y = None, cuda=False, alltoall = False):

        if Y is None:
            Y = X

        if alltoall:
            if cuda:
                return self.cuda_all(X, Y) 
            else: 
                return self.alltoall(X, Y)
        else:
            if len(X) != len(Y):
                return None
            elif cuda:
                return self.cuda_across(X, Y) 
            else: 
                return self.across(X, Y)
            
    def alltoall(self, X, Y):
        pass

    def across(self, X, Y):
        pass

    def cuda_across(self, X, Y):
        self.across(X,Y)

    def cuda_all(self, X, Y):
        self.all(X,Y)



'''def SecondOrder():

    ul = 1.0    #Unit Length
    outer_prod = torch.zeros(self.n, self.m, self.m)
    dist_matrix = torch.zeros(self.m, self.m)
    pdist = nn.allDistance(p=2)
    self.H = torch.zeros(self.n, self.n, dtype=torch.float64)

    #all distances
    for i in range(self.m):
        r_i = torch.tensor([ (i % 4) * ul, int(i / 4) * ul ])
        for j in range(self.m):
            r_j = torch.tensor([ (j % 4) * ul, int(j / 4) * ul ])
            if i != j:
                dist_matrix[i,j] = 1/pdist(r_i , r_j)**2
            else:
                dist_matrix[i,j] = -1

    #2nd order Hamming distance for every 
    for i in range(self.n):
        outer_prod[i] = torch.outer(self.IS_set[i], self.IS_set[i] ) * dist_matrix

        
    #Compute Hamming Distance
    for i in range(self.n):
        for j in range(self.n):
                self.H[i,j] = torch.sum(torch.square(outer_prod[i] - outer_prod[j]))'''