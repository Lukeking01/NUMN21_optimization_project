import numpy as np
from scipy.optimize import minimize_scalar, fmin_bfgs
import matplotlib.pyplot as plt
from chebyquad_problem import chebyquad, gradchebyquad

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

class OptimizationMethod:
    def __init__(self, tol=1e-8, max_iter=1000):
        self.tol = tol
        self.max_iter = max_iter

    def compute_direction(self, problem, x):
        raise NotImplementedError

    def solve(self, problem, x0):
        x = np.asarray(x0, dtype=float)
        history = [x.copy()]

        for k in range(self.max_iter):
            grad = problem.gradient(x)
        
            if np.linalg.norm(grad)<self.tol:
                print(f"Converged in {k} iterations.")
                return x, np.array(history)

            s = self.compute_direction(problem, x)
            x += s
            history.append(x.copy())

        print("Reached maximum iterations without full convergence.")
        return x, np.array(history)

class NewtonMethod(OptimizationMethod):
    def __init__(self, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def compute_direction(self, problem, x0):
        x = np.array(x0, dtype=float)
        grad = problem.gradient(x)
        H = problem.approx_hessian(x)
        s = np.linalg.solve(H,-grad)
        return self.alpha*s

class NewtonMethodLineSearch(NewtonMethod):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_step_size(self, problem, x, dir):
        phi = lambda alpha: problem.func(x+alpha*dir)
        alpha_opt = minimize_scalar(phi, bounds=(0,10), method='bounded').x
        return alpha_opt

    def compute_direction(self, problem, x):
        s = super().compute_direction(problem, x)
        alpha = self.compute_step_size(problem, x, s)
        return alpha*s

class NewtonMethodInexactLineSearch(NewtonMethod):
    def __init__(self, rho=0.01, sigma=0.1, max_alpha=10.0, max_iter=1000, **kwargs):
        super().__init__(**kwargs)
        self.rho = rho
        self.sigma = sigma
        self.max_alpha = max_alpha
        self.max_iter = max_iter

    def inexact_line_search(self, problem, x, s):
        f0 = problem.func(x)
        g0 = problem.gradient(x)
        dphi0 = np.dot(g0, s)

        alpha_prev = 0
        f_prev = f0
        alpha_curr = 1.0
        i = 1

        for _ in range(self.max_iter):
            x_curr = x+alpha_curr*s
            f_curr = problem.func(x_curr)

            if (f_curr>f0+self.rho*alpha_curr*dphi0) or (i>1 and f_curr>=f_prev):
                return zoom(problem, x, s, alpha_prev, alpha_curr, f0, dphi0, self.rho, self.sigma)
        
            g_curr = problem.gradient(x_curr)
            dphi_curr = np.dot(g_curr, s)

            if np.abs(dphi_curr) <= -self.sigma*dphi0:
                return alpha_curr

            if dphi_curr >= 0:
                return zoom(problem, x, s, alpha_curr, alpha_prev, f0, dphi0, self.rho, self.sigma)

            alpha_prev = alpha_curr
            f_prev = f_curr
            alpha_curr = min(2.0*alpha_curr, self.max_alpha)
            i += 1

    def compute_direction(self, problem, x):
        s = super().compute_direction(problem, x)
        alpha = self.inexact_line_search(problem, x, s)
        return alpha*s



def inexact_line_search(func, gradient, x, s, rho=1e-4, sigma=0.9, max_alpha=10.0, max_iter=1000):
    f0 = func(x)
    g0 = gradient(x)
    dphi0 = np.dot(g0, s)

    alpha_prev = 0
    f_prev = f0
    alpha_curr = 1.0
    i = 1

    for _ in range(max_iter):
        x_curr = x+alpha_curr*s
        f_curr = func(x_curr)

        if (f_curr>f0+rho*alpha_curr*dphi0) or (i>1 and f_curr>=f_prev):
            return zoom(func, gradient, x, s, alpha_low=alpha_prev, alpha_high=alpha_curr, f0=f0, dphi0=dphi0, rho=rho, sigma=sigma)
        
        g_curr = gradient(x_curr)
        dphi_curr = np.dot(g_curr, s)

        if np.abs(dphi_curr) <= -sigma*dphi0:
            return alpha_curr

        if dphi_curr >= 0:
            return zoom(func, gradient, x, s, alpha_low=alpha_curr, alpha_high=alpha_prev, f0=f0, dphi0=dphi0, rho=rho, sigma=sigma)

        alpha_prev = alpha_curr
        f_prev = f_curr
        alpha_curr = min(2.0*alpha_curr, max_alpha)
        i += 1

def zoom(func, gradient, x, s, alpha_low, alpha_high, f0, dphi0, rho, sigma, max_iter=1000):
    alpha_j = alpha_low
    for _ in range(max_iter):
        alpha_j = 0.5*(alpha_low+alpha_high)
        x_j = x+alpha_j*s
        f_j = problem.func(x_j)
        f_low = problem.func(x+alpha_low*s)

        if (f_j>f0+rho*alpha_j*dphi0) or (f_j>=f_low):
            alpha_high = alpha_j
        else:
            g_j = problem.gradient(x_j)
            dphi_j = np.dot(g_j, s)

            if np.abs(dphi_j)<=-sigma*dphi0:
                return alpha_j

            if dphi_j*(alpha_high-alpha_low)>=0:
                alpha_high = alpha_low

            alpha_low = alpha_j

    return alpha_j

class QuasiNewtonMethods(NewtonMethod):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def good_broyden(self, H_curr, delta, gamma):
        u = delta-H_curr@gamma
        denom = np.dot(delta, H_curr@gamma)
        return H_curr+np.outer(u, delta@H_curr)/denom

    def bad_broyden(self, H_curr, delta, gamma):
        u = delta-H_curr@gamma
        denom = np.dot(gamma, gamma)
        return H_curr+np.outer(u, gamma)/denom

    def symmetric_broyden(self, H_curr, delta, gamma):
        u = delta-H_curr@gamma
        denom = np.dot(u, gamma)
        if np.abs(denom)<1e-8:
            return H_curr
        return H_curr+np.outer(u, u)/denom

    def DFP(self, H_curr, delta, gamma):
        Hg = H_curr@gamma
        denom1 = np.dot(delta, gamma)
        denom2 = np.dot(gamma, Hg)
        term1 = np.outer(delta, delta)/denom1
        term2 = np.outer(Hg, Hg)/denom2
        return H_curr+term1-term2

    def BFGS(self, H_curr, delta, gamma):
        denom = np.dot(delta, gamma)
        Hg = H_curr@gamma
        gHg = np.dot(gamma, Hg)
        coeff = 1.0+(gHg/denom)
        term1 = coeff*(np.outer(delta, delta)/denom)
        term2 = (np.outer(delta, Hg)+np.outer(Hg, delta))/denom
        return H_curr+term1-term2

class QuasiNewtonSolver(QuasiNewtonMethods):
    def __init__(self, update_method="BFGS", use_line_search=True, rho=0.01, sigma=0.1, **kwargs):
        super().__init__(**kwargs)
        self.update_method = update_method
        self.use_line_search = use_line_search
        self.rho = rho
        self.sigma = sigma
        self.update_functions = {"good_broyden": self.good_broyden,
                                "bad_broyden": self.bad_broyden,
                                "symmetric_broyden": self.symmetric_broyden,
                                "DFP": self.DFP,
                                "BFGS": self.BFGS}

    def solve(self, problem, x0):
        x = np.asarray(x0, dtype=float)
        n = len(x)
        H = np.eye(n)

        update_func = self.update_functions[self.update_method]
        history_x = [x.copy()]
        history_errors = []
        for i in range(self.max_iter):
            g = problem.gradient(x)

            G_true = problem.approx_hessian(x)
            try:
                G_inv_true = np.linalg.inv(G_true)
            except np.linalg.LinAlgError:
                G_inv_true = np.nan * np.ones((n, n))

            #Frobenius norm error
            err = np.linalg.norm(H - G_inv_true, ord="fro")
            history_errors.append(err)
            
            if np.linalg.norm(g)<self.tol:
                print(f"Converged using {self.update_method} in {i} iterations.")
                return x, np.array(history_x), np.array(history_errors)

            s = -H@g

            if self.use_line_search:
                alpha = inexact_line_search(problem.func, problem.gradient, x, s, rho=self.rho, sigma=self.sigma)
            else:
                alpha = 1.0

            delta = alpha*s
            x_next = x+delta
            g_next = problem.gradient(x_next)
            gamma = g_next-g
            H = update_func(H, delta, gamma)
            x = x_next
            history_x.append(x.copy())

        print("Reached maximum iterations.")
        return x, np.array(history_x), np.array(history_errors)

# test Newton

def rosenbrock(x):
    return 100*(x[1]-x[0]**2)**2+(1-x[0])**2

def rosenbrock_grad(x):
    g0 = -400.0 * x[0] * (x[1] - x[0] ** 2) - 2.0 * (1.0 - x[0])
    g1 = 200.0 * (x[1] - x[0] ** 2)
    return np.array([g0, g1])

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

# test inexact line search

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

# solver_bfgs = QuasiNewtonSolver(update_method="BFGS", tol=1e-6)
# x_opt = solver_bfgs.solve(problem, x0=[-1.2, 1.0])
# print(x_opt)

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