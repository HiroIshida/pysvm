import math
import numpy as np
import numpy.linalg as la
import numpy.random as rn

class rbf_kernel:

    def __init__(self, gamma):
        self.gamma = 1.0
    def k(self, x1, x2):
        return math.exp(-self.gamma * la.norm(x1 - x2) ** 2)

def construct_kmat(X, kern):
    n_train = X.shape[0]
    n_dim = X.shape[1]
    mat = np.zeros((n_train, n_train))
    for i in range(n_train):
        for j in range(n_train):
            x1 = X[i, :]
            x2 = X[j, :]
            mat[i, j] = kern.k(x1, x2)
    return mat

def construct_ymat(Y):
    n_train = X.shape[0]
    n_dim = X.shape[1]
    mat = np.zeros((n_train, n_train))
    for i in range(n_train):
        for j in range(n_train):
            y1 = Y[i]
            y2 = Y[j]
            mat[i, j] = y1 * y2
    return mat

def construct_qp_problem(X, Y, kern):
    Kmat = construct_kmat(X, kern)
    Ymat = construct_ymat(Y)
    P = Kmat * Ymat
    return P

    

if __name__=='__main__':
    kern = rbf_kernel(1.0)
    N= 10
    X = rn.randn(N, 3)
    Y = (-1 + (rn.randn(N) > 0)*2)
    P = construct_qp_problem(X, Y, kern)

