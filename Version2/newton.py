import numpy as np
from scipy.optimize import minimize_scalar

from line_search import zoom


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
                return zoom(problem.func, problem.gradient, x, s, alpha_prev, alpha_curr, f0, dphi0, self.rho, self.sigma)
        
            g_curr = problem.gradient(x_curr)
            dphi_curr = np.dot(g_curr, s)

            if np.abs(dphi_curr) <= -self.sigma*dphi0:
                return alpha_curr

            if dphi_curr >= 0:
                return zoom(problem.func, problem.gradient, x, s, alpha_curr, alpha_prev, f0, dphi0, self.rho, self.sigma)

            alpha_prev = alpha_curr
            f_prev = f_curr
            alpha_curr = min(2.0*alpha_curr, self.max_alpha)
            i += 1

    def compute_direction(self, problem, x):
        s = super().compute_direction(problem, x)
        alpha = self.inexact_line_search(problem, x, s)
        return alpha*s
