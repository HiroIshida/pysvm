import math
import numpy as np
import numpy.linalg as la
import numpy.random as rn
import cvxopt

class rbf_kernel:

    def __init__(self, gamma):
        self.gamma = 1.0
    def k(self, x1, x2):
        return math.exp(-self.gamma * la.norm(x1 - x2) ** 2)

class SVM:
    def __init__(self, X, Y, kern, C = 100):
        self.X = X
        self.Y = Y
        self.n_train = X.shape[0]
        self.n_dim = X.shape[1]
        self.kern = kern
        self.C = C

    def solve_qp(self):
        P = np.zeros((self.n_train, self.n_train))
        for i in range(self.n_train):
            for j in range(self.n_train):
                x1 = self.X[i, :]
                x2 = self.X[j, :]
                y1 = self.Y[i]
                y2 = self.Y[j]
                P[i, j] = y1 * y2 * kern.k(x1, x2)

        P = cvxopt.matrix(P)
        q = cvxopt.matrix(np.ones(self.n_train))
        A = cvxopt.matrix(self.Y.astype(np.double), (1, self.n_train))
        b = cvxopt.matrix(0.0)

        G0 = - np.eye(self.n_train)
        G1 = np.eye(self.n_train)
        G = cvxopt.matrix(np.vstack((G0, G1)))
        h0 = np.zeros(self.n_train)
        h1 = np.ones(self.n_train) * self.C
        h = cvxopt.matrix(np.block([h0, h1]))
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        return sol

if __name__=='__main__':
    kern = rbf_kernel(1.0)
    N= 2
    X = rn.randn(N, 3)
    Y = (-1 + (rn.randn(N) > 0)*2)
    svm = SVM(X, Y, kern)
    sol = svm.solve_qp()
