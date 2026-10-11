import numpy as np 
from optimization_problem import OptimizationProblem as opti 

from Rosenbrock_problem import rosenbrock, rosenbrock_gradient, plot_rosenbrock_path

import matplotlib.pyplot as plt

from optimization_problem import NewtonMethod, InexactLineSearch, ExactLineSearch, BFGS, GoodBroyden, BadBroyden, SymmetricBroyden, DFP

rosenbrock_problem = opti(rosenbrock,rosenbrock_gradient)
Newton = NewtonMethod(rosenbrock_problem)
Newton_Inexact_line_search = NewtonMethod(rosenbrock_problem, line_search = InexactLineSearch())
Newton_Exact_line_search = NewtonMethod(rosenbrock_problem, line_search = ExactLineSearch())
BFGS_method = BFGS(rosenbrock_problem)
GoodBroyden_method = GoodBroyden(rosenbrock_problem)
BadBroyden_method = BadBroyden(rosenbrock_problem)
SymmetricBroyden_method = SymmetricBroyden(rosenbrock_problem)
DFP_method = DFP(rosenbrock_problem)

methods = np.array([Newton, 
                    Newton_Exact_line_search,
                    Newton_Inexact_line_search,
                    BFGS_method,
                    GoodBroyden_method,
                    BadBroyden_method,
                    SymmetricBroyden_method,
                    DFP_method])


num_cols = 4
num_rows = int(np.ceil(len(methods)/num_cols))

fig, axes = plt.subplots(num_rows, 
                         num_cols,
                         figsize =(4*num_cols,4*num_rows),
                         squeeze=False)
for i, method in enumerate(methods):
    initial_guess = (0,0)
    method.minimize(initial_guess)
    row = i // num_cols
    col = i % num_cols
    plot_rosenbrock_path(methods[i], axes[row,col])

plt.tight_layout()
plt.show()