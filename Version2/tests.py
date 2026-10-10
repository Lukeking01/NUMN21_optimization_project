import numpy as np
from scipy.optimize import fmin_bfgs
import matplotlib.pyplot as plt
from chebyquad_problem import chebyquad, gradchebyquad

from problem import OptimizationProblem
from newton import NewtonMethodLineSearch, NewtonMethodInexactLineSearch
from quasi_newton import QuasiNewtonSolver
from line_search import inexact_line_search
from test_functions import rosenbrock, rosenbrock_grad

# ---------------------------------------------------------------
# test Newton
# ---------------------------------------------------------------

# x0 = [-1.2, 1.0]
# problem = OptimizationProblem(rosenbrock)
# solver = NewtonMethodLineSearch(tol=1e-6)
# x_opt, history = solver.solve(problem, x0)
# print(x_opt)

# make the contour plot

# x_min, x_max = history[:, 0].min() - 0.5, history[:, 0].max() + 0.5
# y_min, y_max = history[:, 1].min() - 0.5, history[:, 1].max() + 0.5

# X, Y = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
# Z = np.array([rosenbrock([x, y]) for x, y in zip(X.ravel(), Y.ravel())]).reshape(X.shape)

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# levels = np.logspace(-1, 3, 20)
# CS = ax1.contour(X, Y, Z, levels=levels)
# ax1.clabel(CS, inline=True, fontsize=8)
# ax1.set_title("Objective Function Contour Plot")
# ax1.set_xlabel("$x$")
# ax1.set_ylabel("$y$")
# ax1.grid(True, linestyle="--", alpha=0.5)

# ax2.contour(X, Y, Z, levels=levels)

# ax2.plot(history[:, 0], history[:, 1], "r--o", linewidth=1.5, markersize=5, label="Newton Path")

# # Highlight starting point (x0) and optimal point (x*)
# ax2.plot(history[0, 0], history[0, 1], "go", markersize=8, label="Start $x_0$")
# ax2.plot(x_opt[0], x_opt[1], "r*", markersize=12, label="Minimum $x^*$")

# ax2.set_title("Optimization Steps Towards Minimum")
# ax2.set_xlabel("$x$")
# ax2.set_ylabel("$y$")
# ax2.legend()
# ax2.grid(True, linestyle="--", alpha=0.5)

# plt.tight_layout()
# plt.show()

# ---------------------------------------------------------------
# test inexact line search
# ---------------------------------------------------------------

# x0 = np.array([0.0, 0.0])
# s0 = np.array([1.0, 0.0])

# rho_param = 0.01
# sigma_param = 0.1

# alpha_opt = inexact_line_search(rosenbrock, rosenbrock_grad, x0, s0, rho_param, sigma_param)
# print(alpha_opt)


# x0 = [-1.2, 1.0]
# problem = OptimizationProblem(rosenbrock)
# solver_1 = NewtonMethodLineSearch(tol=1e-6)
# x_opt_1, history_1 = solver_1.solve(problem, x0)
# print(x_opt_1)

# solver_2 = NewtonMethodInexactLineSearch(rho=0.01, sigma=0.1, tol=1e-6)
# x_opt_2, history_2 = solver_2.solve(problem, x0)
# print(x_opt_2)

# ---------------------------------------------------------------
# test quasi-Newton (BFGS)
# ---------------------------------------------------------------

# solver_bfgs = QuasiNewtonSolver(update_method="BFGS", tol=1e-6)
# x_opt = solver_bfgs.solve(problem, x0=[-1.2, 1.0])
# print(x_opt)

# ---------------------------------------------------------------
# test chebyquad
# ---------------------------------------------------------------

# problem = OptimizationProblem(chebyquad, gradchebyquad)
# dim = [4, 8, 11]

# for n in dim:
#     print(f"Running chebyquad for n={n}")
#     x0 = np.linspace(0, 1, n)
#     solver = QuasiNewtonSolver(update_method="BFGS", tol=1e-6, max_iter=2000)
#     x_opt = solver.solve(problem, x0)
#     print(f"Final values: {chebyquad(x_opt):.2e}")

#     x_opt_scipy = fmin_bfgs(chebyquad, x0, gradchebyquad, disp=False)
#     print(f"SciPy BFGS Final values:  {chebyquad(x_opt_scipy):.2e}")
