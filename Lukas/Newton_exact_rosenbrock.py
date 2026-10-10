import numpy as np 
from optimization_problem import OptimizationProblem as opti 

from Rosenbrock_problem import rosenbrock, rosenbrock_gradient, plot_rosenbrock_path

import matplotlib.pyplot as plt

from optimization_problem import NewtonMethod, ExactLineSearch

rosenbrock_problem = opti(rosenbrock,rosenbrock_gradient)
Newton = NewtonMethod(rosenbrock_problem)
Newton_Exact_line_search = NewtonMethod(rosenbrock_problem, line_search = ExactLineSearch())

methods = np.array([Newton, Newton_Exact_line_search])



num_cols = 2
num_rows = int(np.ceil(len(methods)/num_cols))

fig, axes = plt.subplots(num_rows, 
                         num_cols,
                         figsize =(16,8),
                         squeeze=False)
for i, method in enumerate(methods):
    initial_guess = (0,0)
    method.minimize(initial_guess)
    row = i // num_cols
    col = i % num_cols
    plot_rosenbrock_path(methods[i], axes[row,col])

plt.show()
