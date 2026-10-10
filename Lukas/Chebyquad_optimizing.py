
from chebyquad_problem import chebyquad, gradchebyquad
import numpy as np
import matplotlib.pyplot as plt
import optimization_problem as op
from optimization_problem import NewtonMethod, InexactLineSearch, ExactLineSearch, BFGS, GoodBroyden, BadBroyden, SymmetricBroyden, DFP

import scipy.optimize as fbg

cheb_problem = op.OptimizationProblem(chebyquad, gradchebyquad)

Newton = NewtonMethod(cheb_problem)
Newton_Inexact_line_search = NewtonMethod(cheb_problem, line_search = InexactLineSearch())
Newton_Exact_line_search = NewtonMethod(cheb_problem, line_search = ExactLineSearch())
BFGS_method = BFGS(cheb_problem)
GoodBroyden_method = GoodBroyden(cheb_problem)
BadBroyden_method = BadBroyden(cheb_problem)
SymmetricBroyden_method = SymmetricBroyden(cheb_problem)
DFP_method = DFP(cheb_problem)

methods = np.array([
                    # Newton, 
                    # Newton_Exact_line_search,
                    # Newton_Inexact_line_search,
                    BFGS_method,
                    GoodBroyden_method,
                    BadBroyden_method,
                    SymmetricBroyden_method,
                    DFP_method])

for method in methods:
    print(f"===== Method: {method.__class__.__name__} =====")
    for dim in [4,8,11]:
        initial_guess = np.random.rand(dim)
        method.minimize(initial_guess)
        final = method.history[-1]
        scipy_result = fbg.fmin_bfgs(chebyquad,initial_guess, gradchebyquad, disp=False)
        difference = final - scipy_result
        print(f"Dimension: {dim}, Iterations: {len(method.history)}, \n Final point: {final}, \n Scipy says: {scipy_result}, \n Difference: {difference}")
    