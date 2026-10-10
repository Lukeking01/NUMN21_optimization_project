
import numpy as np 
from optimization_problem import OptimizationProblem as opti 

import matplotlib.pyplot as plt

from optimization_problem import NewtonMethod, ExactLineSearch, InexactLineSearch

from Rosenbrock_problem import rosenbrock, rosenbrock_gradient, plot_rosenbrock_path

problem = opti(rosenbrock,rosenbrock_gradient)

inexact = InexactLineSearch()
exact = ExactLineSearch()

x0 = np.array([-1.2, 1.0])              
p0 = -problem.gradient_at(x0)

print(inexact.search(problem, x0, p0))
print(exact.search(problem, x0, p0))