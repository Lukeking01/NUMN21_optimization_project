import numpy as np

from newton import NewtonMethod
from line_search import inexact_line_search


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
