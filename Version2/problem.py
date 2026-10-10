import numpy as np


class OptimizationProblem:
    def __init__(self, func, grad = None):
        self.func = func
        self.grad = grad

    def approx_grad(self, x, h=1e-8):
        x = np.asarray(x, dtype=float)
        n = len(x)
        grad = np.zeros(n)
        for i in range(n):
            e = np.zeros(n)
            e[i] = 1.0
            grad[i] = (self.func(x+h*e)-self.func(x-h*e))/(2.0*h)
        return grad

    def gradient(self, x):
        if self.grad is not None:
            return np.asarray(self.grad(x), dtype=float)
        return self.approx_grad(x)

    def approx_hessian(self, x, h=1e-8):
        x = np.asarray(x, dtype=float)
        n = len(x)
        H = np.zeros((n,n), dtype=float)
        for j in range(n):
            e = np.zeros(n)
            e[j] = 1.0
            H[:, j] = (self.gradient(x+h*e)-self.gradient(x-h*e))/(2*h)
        
        return 0.5*(H+H.T)
