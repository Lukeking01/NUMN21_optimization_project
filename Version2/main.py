import numpy as np
import matplotlib.pyplot as plt

from problem import OptimizationProblem
from quasi_newton import QuasiNewtonSolver
from test_functions import rosenbrock, rosenbrock_grad


problem = OptimizationProblem(func=rosenbrock, grad=rosenbrock_grad)
x0 = np.array([-1.2, 1.0])

solver = QuasiNewtonSolver(update_method="BFGS", tol=1e-6)
x_opt, history_x, errors = solver.solve(problem, x0)

plt.figure(figsize=(8, 5))
plt.semilogy(errors, "bo-", linewidth=2, markersize=6)
plt.title("BFGS Approximation Quality")
plt.xlabel("Iteration $k$")
plt.ylabel("Frobenius Norm Error")
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.show()
