
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import optimization_problem as op
from optimization_problem import BFGS, NewtonMethod

def f(x):
    return x**4 - 2*x**2 + 1

def grad(x):
    return 2*x

xs = np.linspace(-2,2,100)
ys = f(xs)

plt.plot(xs,ys)
plt.plot(xs, xs*0)
plt.show()