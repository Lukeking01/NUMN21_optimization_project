import numpy as np


def rosenbrock(x):
    return 100*(x[1]-x[0]**2)**2+(1-x[0])**2

def rosenbrock_grad(x):
    g0 = -400.0 * x[0] * (x[1] - x[0] ** 2) - 2.0 * (1.0 - x[0])
    g1 = 200.0 * (x[1] - x[0] ** 2)
    return np.array([g0, g1])
