import math
import numpy as np
import numpy.linalg as la
import numpy.random as rn
import matplotlib.pyplot as plt
import cvxopt
import time

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

        # dual valiable
        self.a = None

        # parameters of hyperplane
        self.A = None
        self.b = None

    def show(self):
        idx_positive = np.where(svm.Y > 0)[0]
        idx_negative = np.where(svm.Y < 0)[0]
        fig, ax = plt.subplots() 
        ax.scatter(self.X[idx_positive, 0], self.X[idx_positive, 1], c = "r")
        ax.scatter(self.X[idx_negative, 0], self.X[idx_negative, 1], c = "b")

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
        q = cvxopt.matrix(- np.ones(self.n_train))
        A = cvxopt.matrix(self.Y.astype(np.double), (1, self.n_train))
        b = cvxopt.matrix(0.0)

        G0 = - np.eye(self.n_train)
        G1 = np.eye(self.n_train)
        G = cvxopt.matrix(np.vstack((G0, G1)))
        h0 = np.zeros(self.n_train)
        h1 = np.ones(self.n_train) * self.C
        h = cvxopt.matrix(np.block([h0, h1]))
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        self.a = np.array(sol["x"])
        self._compute_hyperplane()

    def _compute_hyperplane(self): 
        ay = self.a * np.array([self.Y]).T
        ayx = np.tile(ay, (1, self.n_dim)) * self.X
        A = ayx.mean(axis = 0)
        i = 2

        b_list = np.array([self.Y[i] - A.dot(self.X[i, :]) for i in range(self.n_train)])
        b = b_list.mean()

        self.A = A
        self.b = b

    def predict(self, X):
        val = np.dot(self.A, X.T) + self.b
        return (val > 0)


def gen_dataset(N = 10):
    X = rn.randn(N, 2) * 10
    logical = np.array([(X[i, 1] - X[i, 0] * 2 < 0) for i in range(N)])
    Y = (-1 + logical * 2)
    return X, Y

if __name__=='__main__':
    #rn.seed(2)
    kern = rbf_kernel(1.0)
    X, Y = gen_dataset(10)
    #X = rn.randn(N, 3)3
    #Y = (-1 + (rn.randn(N) > 0)*2)
    svm = SVM(X, Y, kern)
    svm.solve_qp()
    pre = svm.predict(X)
    logical = (-1 + pre * 2) 
    print logical - Y
    print svm.a

