import scipy
import numpy as np
import torch
from .metric import TensorOperator

from ... import generators


class Euclidean(TensorOperator):

    def all(self, X, Y):
        return scipy.spatial.distance.cdist(X, Y, metric='euclidean')

    def across(self, X, Y):
        return scipy.spatial.distance.cdist(X, Y, metric='euclidean')
        

class IndependentSet(TensorOperator):

    def all(self, ISset, LatticePos=None):

        if LatticePos is None or ( (ISset.shape == LatticePos.shape) and (ISset == LatticePos).all() ):
            lgen = generators.LatticeGenerator()
            LatticePos = lgen.genApproxSquare(ISset.shape[1])

        #Compute Euclidean distance between lattice site
        eMetric = Euclidean()
        dist_matrix = eMetric(X = LatticePos)

        #Compute reciprical distance squared
        non_zero_idx = np.nonzero(dist_matrix)
        dist_matrix[non_zero_idx] = np.square(np.reciprocal(dist_matrix[non_zero_idx]))
        dist_matrix = np.expand_dims(dist_matrix, axis=0)

        #Second Order Hamming
        outerProd = OuterProduct()
        h2_ISSet = outerProd(ISset)

        h2xJ = np.multiply(dist_matrix, h2_ISSet)   # n X m X m

        h2xJ_0 = np.expand_dims(h2xJ, axis=0)
        h2xJ_1 = np.expand_dims(h2xJ, axis=1)

        h2xJ_square = np.square(h2xJ_0 - h2xJ_1)
        

        return np.sum(h2xJ_square, axis=(2,3))

    def across(self, ISset, LatticePos=None):
        return self.all(ISset, LatticePos)


class OuterProduct(TensorOperator):

    def across(self, X, Y):

        if np.array(X).ndim == 1:
            outer_prod = np.outer(X, Y)
        else: 
            #2nd order Hamming distance for every 
            outer_prod = np.zeros([len(X), len(X[0]), len(X[0])])

            for i in range(len(X)):
                outer_prod[i] = np.outer(X[i], Y[i])

        return outer_prod

    def alltoall(self, X, Y):

        #2nd order Hamming distance for every 
        outer_prod = np.zeros([len(X), len(Y), len(X[0]), len(Y[0])])

        for i in range(len(X)):
            for j in range(len(Y)):
                outer_prod[i, j] = np.outer(X[i], Y[j])

        return outer_prod 


class TensorMSE(TensorOperator):

    def across(self, X, Y):
        H = np.zeros([len(X)])

        #Compute Hamming Distance
        for i in range(len(X)):
            H[i] = np.sum(np.square(X[i] - Y[i]))

        return H

    def all(self, X, Y):

        #2nd order Hamming distance for every 
        H = np.zeros([len(X), len(Y)])

        for i in range(len(X)):
            for j in range(len(Y)):
                H[i, j] = np.sum(np.square(X[i] - Y[j]))

        return H 


def unit_test():
    print("Metrics Unit Test")
    print("allEuclidean")

    euclidean_metric = Euclidean()

    t1pE = (euclidean_metric([[0,0], [0,0]]) == [[0,0],[0,0]]).all()
    t2pE = euclidean_metric([[0,1], [1,1]])
    t3pE = euclidean_metric([[0,1], [1,2], [0,2]])

    outer_product = OuterProduct()

    t1_2ndO = outer_product([0,0], [0,0])
    #print(t1_2ndO)
    t2_2ndO = outer_product([0,1], [1,0])
    #print(t2_2ndO)

    iSMetric = IndependentSet()

    print(iSMetric(np.array([[1,0,0], [0,1,0], [0,0,1]])))
    print(iSMetric(np.array([[1,0,0], [0,1,0], [0,0,1]]), np.array([[0,1], [1,2], [0,2]])))
    print(iSMetric(np.array([[1,1,0], [0,1,1], [1,0,1]]), np.array([[0,1], [1,2], [0,2]])))


if __name__ == "__main__":
    #unit_test()
    pass