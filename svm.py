import math
import numpy as np
import numpy.linalg as la
import numpy.random as rn

class rbf_kernel:

    def __init__(self, gamma):
        self.gamma = 1.0
    def k(self, x1, x2):
        return math.exp(-self.gamma * la.norm(x1 - x2) ** 2)

class SVM:
    def __init__(self, X, Y, kern):
        self.X = X
        self.Y = Y
        self.n_train = X.shape[0]
        self.n_dim = X.shape[1]
        self.kern = kern

    def construct_qp_problem(self):
        P = np.zeros((self.n_train, self.n_train))
        for i in range(self.n_train):
            for j in range(self.n_train):
                x1 = self.X[i, :]
                x2 = self.X[j, :]
                y1 = self.Y[i]
                y2 = self.Y[j]
                P[i, j] = y1 * y2 * kern.k(x1, x2)

        q = np.ones(self.n_train)
        A = np.matrix(self.Y)
        b = 0.0

    

if __name__=='__main__':
    kern = rbf_kernel(1.0)
    N= 10
    X = rn.randn(N, 3)
    Y = (-1 + (rn.randn(N) > 0)*2)
    svm = SVM(X, Y, kern)
    P = svm.construct_qp_problem()
    #AP = construct_qp_problem(X, Y, kern)

