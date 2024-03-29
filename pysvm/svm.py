import math
import numpy as np
import numpy.linalg as la
import numpy.random as rn
import matplotlib.pyplot as plt
import cvxopt
import time
import utils

from sklearn.metrics.pairwise import *

eps = 1e-7

class SVM:
    def __init__(self, X, Y, kern, C = 100000.0):
        self.X = X
        self.Y = Y
        self.n_train = X.shape[0]
        self.n_dim = X.shape[1]
        self.kern = kern
        self.C = C

        self.w = None
        self.b = None
        self.support_vector = None

    def show(self):

        fig, ax = plt.subplots() 
        def fun(x):
            f = self.decision_function(np.array([x]))
            return f[0]

        bmin, bmax = svm.get_boudnary()
        utils.show2d(fun, bmin, bmax, levels = [0.0], fax = (fig, ax), N = 100)

        idx_positive = np.where(self.Y > 0)[0]
        idx_negative = np.where(self.Y < 0)[0]
        ax.scatter(self.X[idx_positive, 0], self.X[idx_positive, 1], c = "r")
        ax.scatter(self.X[idx_negative, 0], self.X[idx_negative, 1], c = "b")

    def solve_qp(self):
        gram_matrix = self.kern(self.X)
        T = np.array([[self.Y[i] * self.Y[j] for j in range(self.n_train)] for i in range(self.n_train)])
        P = gram_matrix * T

        P = cvxopt.matrix(P)
        q = cvxopt.matrix(- np.ones(self.n_train))
        A = cvxopt.matrix(self.Y.astype(np.double), (1, self.n_train))
        b = cvxopt.matrix(0.0)

        G0 = np.eye(self.n_train)
        G1 = - np.eye(self.n_train)
        G = cvxopt.matrix(np.vstack((G0, G1)))

        h0 = np.ones(self.n_train) * self.C
        h1 = np.zeros(self.n_train)
        h = cvxopt.matrix(np.block([h0, h1]))
        sol = cvxopt.solvers.qp(P, q, G=G, h=h, A=A, b=b)

        indexes_active = list(filter(lambda x: sol["x"][x] > eps, range(self.n_train)))
        self.w = np.array(sol["x"])[indexes_active].reshape(len(indexes_active)) * self.Y[indexes_active]
        self.support_vector = self.X[indexes_active]

        tmp_list = []
        for i in indexes_active:
            tmp = 0
            for j in indexes_active:
                tmp += (sol["x"][j] * self.Y[j] * gram_matrix[i][j])
            tmp_list.append(self.Y[i]-tmp)
        self.b = np.mean(tmp_list)

    def decision_function(self, X):
        y = np.dot(np.array([self.w]), self.kern(X, Y=self.support_vector).T) + self.b
        y = y.reshape(len(X))
        return y

    def predict(self, X):
        y = self.decision_function(X)
        return np.array([1 if pred > 0 else -1 for pred in y])

    def get_boudnary(self, margin = 0.2):
        bmin_ = self.X.min(axis = 0) 
        bmax_ = self.X.max(axis = 0) 
        dif = bmax_ - bmin_
        bmin = bmin_ - margin * dif
        bmax = bmax_ + margin * dif
        return bmin, bmax

def gen_dataset_linear(N = 10):
    X = rn.randn(N, 2) * 10
    logical = np.array([(X[i, 1] - X[i, 0] * 2 < 0) for i in range(N)])
    Y = (-1 + logical * 2)
    return X, Y

def gen_dataset_circle(N = 30):
    X = rn.randn(N, 2) * 1
    logical = np.array([(X[i, 1] ** 2 + X[i, 0] ** 2 < 1.0) for i in range(N)])
    Y = (-1 + logical * 2)
    return X, Y

if __name__=='__main__':
    np.random.seed(0)
    X, Y = gen_dataset_circle(N = 100)
    kern = rbf_kernel
    svm = SVM(X, Y, kern)
    svm.solve_qp()
    svm.show()
    plt.show()
