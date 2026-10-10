import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# TASK 1 – Optimization problem
# ============================================================

class OptimizationProblem:
    def __init__(self, objective, gradient=None):
        self.objective = objective
        self.gradient = gradient

    def evaluate(self, x):
        return self.objective(x)

    def gradient_at(self, x):
        if self.gradient is None:
            raise ValueError("No gradient was provided.")
        return self.gradient(x)


# ============================================================
# TASK 2 – General optimization method
# ============================================================

class OptimizationMethod:
    def __init__(self, problem: OptimizationProblem, tolerance: float=1e-6, max_iterations: int=1000):
        self.problem = problem
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.history = []
        self.function_values = []

    def minimize(self, x0):
        raise NotImplementedError

    def reset_history(self):
        self.history = []
        self.function_values = []

    def store(self, x):
        self.history.append(x.copy())
        self.function_values.append(self.problem.evaluate(x))


# ============================================================
# Utility: finite-difference Hessian
# ============================================================

def finite_difference_hessian(problem, x, h=1e-5):
    '''
    Computes the Hessian matrix, G, of the objective function at point x using forward finite differences
    '''
    n = len(x)
    G = np.zeros((n, n))
    g0 = problem.gradient_at(x)

    for i in range(n):
        x_forward = x.copy()
        x_forward[i] += h
        G[:, i] = (problem.gradient_at(x_forward) - g0) / h

    return 0.5 * (G + G.T)          # symmetrise


# ============================================================
# TASK 3 – Classical Newton
# ============================================================

class NewtonMethod(OptimizationMethod):
    def __init__(self, problem, tolerance=1e-6, max_iterations=1000, hessian_step=1e-5, line_search=None):
        
        super().__init__(problem, tolerance, max_iterations)
        
        self.line_search = line_search
        self.hessian_step = hessian_step

    def get_hessian(self, x):
        return finite_difference_hessian(self.problem, x, self.hessian_step)

    def minimize(self, x0):
        self.reset_history()
        
        x = np.asarray(x0, dtype=float).copy()
        self.store(x)

        for _ in range(self.max_iterations):
            g = self.problem.gradient_at(x)
            if np.linalg.norm(g) < self.tolerance: # Residual criterion
                break
            G = self.get_hessian(x)
            s = np.linalg.solve(G, -g)
            if self.line_search is not None:
                alpha = self.line_search.search(self.problem, x, s)
                x = x + alpha * s
            else:
                x = x + s
            self.store(x)
        return x


# ============================================================
# TASK 4 – Exact line search 
# ============================================================

class ExactLineSearch:
    def __init__(self, tolerance=1e-8, max_step=1.0, max_iter=100):
        self.tolerance = tolerance
        self.max_step = max_step
        self.max_iter = max_iter

    def search(self, problem, x, p):
        invphi = (np.sqrt(5) - 1) / 2
        invphi2 = (3 - np.sqrt(5)) / 2

        a, b = 0.0, self.max_step

        c = a + invphi2 * (b - a)
        d = a + invphi * (b - a)

        yc = problem.evaluate(x + c * p)
        yd = problem.evaluate(x + d * p)

        iterations = 0

        while (b - a) > self.tolerance and iterations < self.max_iter:
            if yc < yd:
                b = d
                d, yd = c, yc
                c = a + invphi2 * (b - a)
                yc = problem.evaluate(x + c * p)
            else:
                a = c
                c, yc = d, yd
                d = a + invphi * (b - a)
                yd = problem.evaluate(x + d * p)

            iterations += 1

        if (b - a) > self.tolerance:
            raise RuntimeError(
                "Line search did not reach the requested tolerance"
            )

        return (a + b) / 2



# ============================================================
# TASKS 6-7 – Strong Wolfe line search
# ============================================================

class InexactLineSearch:
    """
    Inexact line search based on the Goldstein/Wolfe conditions
    """

    def __init__(self, rho=0.01, sigma=0.1,
                 tau1=9.0, tau2=0.05, tau3=0.5,
                 max_bracket=50, max_section=50):
        assert 0 < rho < sigma < 1
        assert 0 < tau2 < tau3 <= 0.5
        self.rho = rho
        self.sigma = sigma
        self.tau1 = tau1          # extrapolation factor
        self.tau2 = tau2          # sectioning lower safeguard
        self.tau3 = tau3          # sectioning upper safeguard
        self.max_bracket = max_bracket
        self.max_section = max_section

    def search(self, problem, x, p, alpha0=1.0):
        """
        Return a step length α > 0 that satisfies the strong Wolfe conditions.
        """
        f0 = problem.evaluate(x)
        g0 = problem.gradient_at(x)
        dphi0 = np.dot(g0, p)

        if dphi0 >= 0:
            raise ValueError("Search direction is not a descent direction")

        # ------------------------------------------------------------------
        # Bracketing phase (Fletcher 2.6.2 – with the corrected test)
        # ------------------------------------------------------------------
        alpha_prev = 0.0
        f_prev = f0
        dphi_prev = dphi0
        alpha = alpha0

        for _ in range(self.max_bracket):
            f = problem.evaluate(x + alpha * p)

            # corrected test from the project note
            if f > f0 + self.rho * alpha * dphi0 or f >= f_prev:
                # bracket found: [alpha_prev, alpha]
                return self._section(problem, x, p,
                                     alpha_prev, alpha,
                                     f0, dphi0,
                                     f_prev, dphi_prev, f)

            dphi = np.dot(problem.gradient_at(x + alpha * p), p)

            # curvature condition already satisfied → accept
            if abs(dphi) <= -self.sigma * dphi0:
                return alpha

            if dphi >= 0:
                # bracket found: [alpha, alpha_prev]  (order does not matter)
                return self._section(problem, x, p,
                                     alpha, alpha_prev,
                                     f0, dphi0,
                                     f, dphi, f_prev)

            # still decreasing – extrapolate
            alpha_next = alpha + self.tau1 * (alpha - alpha_prev)
            alpha_prev, f_prev, dphi_prev = alpha, f, dphi
            alpha = alpha_next

        # failed to bracket – return the last trial point
        return alpha

    def _section(self, problem, x, p, a, b, f0, dphi0, fa, dphia, fb):
        """
        Sectioning phase (Fletcher 2.6.4).
        On entry [a,b] brackets an acceptable point.
        """
        for _ in range(self.max_section):
            # safeguard: pick trial point inside (a,b) away from the ends
            # (the precise interpolant is secondary; bisection is fine)
            alpha = 0.5 * (a + b)
            # optional: force the safeguards of Fletcher
            # alpha = a + np.clip(alpha-a, self.tau2*(b-a), self.tau3*(b-a))

            f = problem.evaluate(x + alpha * p)

            if f > f0 + self.rho * alpha * dphi0 or f >= fa:
                # shrink from the right
                b = alpha
                fb = f
            else:
                dphi = np.dot(problem.gradient_at(x + alpha * p), p)

                # strong Wolfe curvature satisfied → accept
                if abs(dphi) <= -self.sigma * dphi0:
                    return alpha

                if dphi * (b - a) >= 0:
                    b = a
                    fb = fa

                a = alpha
                fa = f
                dphia = dphi

        # max iterations reached – return the best point we have
        return a


# ============================================================
# TASK 9 – Quasi-Newton base
# ============================================================

class QuasiNewtonMethod(OptimizationMethod):
    def __init__(self, problem, tolerance=1e-6, max_iterations=1000, line_search=None):
        super().__init__(problem, tolerance, max_iterations)
        self.line_search = line_search or InexactLineSearch()
        self.H = None          # inverse Hessian approx
        self.G = None          # Hessian approx

    def initialize_matrices(self, x):
        n = len(x)
        self.G = np.eye(n)
        self.H = np.eye(n)

    def direction(self, g):
        p = -self.H @ g
        # safeguard: if the quasi-Newton direction is not a descent direction,
        # fall back to steepest descent
        if np.dot(g, p) >= 0:
            p = -g
        return p

    def update(self, s, y):
        raise NotImplementedError

    def minimize(self, x0):
        self.reset_history()
        x = np.asarray(x0, dtype=float).copy()
        self.initialize_matrices(x)
        self.store(x)
        first = True

        for _ in range(self.max_iterations):
            g = self.problem.gradient_at(x)
            if np.linalg.norm(g) < self.tolerance:
                break
            p = self.direction(g)
            alpha = self.line_search.search(self.problem, x, p)
            x_new = x + alpha * p
            g_new = self.problem.gradient_at(x_new)
            s = x_new - x
            y = g_new - g
            if first and np.dot(s, y) > 0:
                # scale H₀ = (sᵀy / yᵀy) I
                self.H = (np.dot(s, y) / np.dot(y, y)) * np.eye(len(s))
                first = False
            self.update(s, y)
            x = x_new
            self.store(x)
        return x


# ============================================================
# Good Broyden
# ============================================================

class GoodBroyden(QuasiNewtonMethod):
    def update(self, s, y):
        if np.dot(s, y) <= 1e-14:
            return
        Gs = self.G @ s
        self.G = self.G + np.outer(y - Gs, s) / np.dot(s, s)

        u = y - Gs
        v = s / np.dot(s, s)
        den = 1.0 + v @ self.H @ u
        if abs(den) > 1e-14:
            self.H = self.H - np.outer(self.H @ u, v @ self.H) / den
        else:
            self.H = np.linalg.pinv(self.G)


# ============================================================
# Bad Broyden
# ============================================================

class BadBroyden(QuasiNewtonMethod):
    def update(self, s, y):
        Hy = self.H @ y
        den = np.dot(y, y)
        if den < 1e-14:
            return
        self.H = self.H + np.outer(s - Hy, y) / den
        self.G = np.linalg.pinv(self.H)


# ============================================================
# Symmetric Broyden
# ============================================================

class SymmetricBroyden(QuasiNewtonMethod):
    def update(self, s, y):
        r = y - self.G @ s
        sTs = np.dot(s, s)
        if sTs < 1e-14:
            return
        self.G = (self.G
                  + (np.outer(r, s) + np.outer(s, r)) / sTs
                  - (np.dot(r, s) / sTs**2) * np.outer(s, s))
        self.G = 0.5 * (self.G + self.G.T)
        self.H = np.linalg.pinv(self.G)


# ============================================================
# DFP
# ============================================================

class DFP(QuasiNewtonMethod):
    def update(self, s, y):
        sy = np.dot(s, y)
        Hy = self.H @ y
        yHy = np.dot(y, Hy)
        if abs(sy) < 1e-14 or abs(yHy) < 1e-14:
            return
        self.H = self.H + np.outer(s, s) / sy - np.outer(Hy, Hy) / yHy
        self.H = 0.5 * (self.H + self.H.T)
        self.G = np.linalg.pinv(self.H)


# ============================================================
# BFGS
# ============================================================

class BFGS(QuasiNewtonMethod):
    def update(self, s, y):
        sy = np.dot(s, y)
        if sy <= 1e-14:
            return
        rho = 1.0 / sy
        I = np.eye(len(s))
        V = I - rho * np.outer(s, y)
        self.H = V @ self.H @ V.T + rho * np.outer(s, s)
        self.H = 0.5 * (self.H + self.H.T)
        self.G = np.linalg.pinv(self.H)