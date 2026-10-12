import numpy as np
import scipy.linalg as linalg
import scipy.differentiate as differentiate
from line_search import exact_line_search, arm_wolf_search, line_search_alt_orsolya
from hessian_approx import finite_diffs

class Problem:
    def __init__(self, objective, gradient=None):
        self.objective = objective
        if gradient != None:
            self.gradient = gradient
        else:
            self.gradient = self.approx_grad()

    def evaluate(self, x):
        return self.objective(x)

    def gradient_at(self, x):
        if self.gradient is not None:
            return np.asarray(self.gradient(x), dtype=float)
        return self.approx_grad(x)

    def approx_grad(self, h=1e-8):
        def gradient(x):
            x = np.asarray(x, dtype=float)
            n = len(x)
            grad = np.zeros(n)
            for i in range(n):
                e = np.zeros(n)
                e[i] = 1.0
                grad[i] = (self.objective(x+h*e)-self.objective(x-h*e))/(2.0*h)
            return grad
        return gradient

class Optimization():
    """
    Optimization problem class, implements a "reset" which is run internally on each "solve"
    and implements the main solving steps for approaching a minimum of the objective function.

    Subclasses that implement the chosen optimization approach should override the line_search
    and approx_hessian functions, and the approx_hessian function should specify if it is returning
    an approximation of the Hessian itself, or its inverse.
    """

    def __init__(self, problem):
        # Objective function and gradient
        self.problem = problem
        # Stopping criteria # TODO Move to config
        self.grad_tol = 1e-6
        self.step_tol = 1e-6
        self.max_steps = 10_000
        # Eval history
        self.step_history = []
        self.error_history = []
        # TODO Temp fields
        self.phi_prime = []

    def reset(self):
        """
        To be overriden in subclasses that require a reset between solving problems.
        """
        self.step_history = []
        self.error_history = []
        self.phi_prime = []

    def line_search(self, x, s):
        """
        Abstract method - to be overriden by subclass.

        Should return a scalar "alpha" which will be the step size.
        """
        raise NotImplementedError("Subclass must provide an implementation!")

    def approx_hessian(self, g, x):
        """
        Abstract method - to be overriden by subclass.

        Should return a symmetric matrix to serve as the Hessian or Hessian inverse.
        Return also a flag "is_inv" set to True if we are approximating the Hessian inverse
        directly.
        """
        raise NotImplementedError("Subclass must provide an implementation!")

    def solve(self, x0):
        """
        Run through iterations of solving an optimization problem.

        Terms:
            - Gradient:         grad (row-vector)
            - Hessian:          hess (matrix)
            - Hessian Inv:      hess_inv (matrix)
            - Step Dir:         sdir (row-vector)
            - Step Size:        alpha (scalar)
            - x on k-th iter:   x_k (scalar)

        Iteration Flow: (Choose initial guess)
            1) Approximate the Hessian and solve Hs = -g
            2) Use line search to find a step size alpha
            3) Update the current value of x_k
            4) Check stopping conditions (Residual/Cauchy)
        """

        self.reset()

        x = np.array(x0)
        self.step_history = []
        
        for _ in range(self.max_steps):
            g = self.problem.gradient_at(x)

            # Check exit criteria
            if np.linalg.norm(g) < self.grad_tol:
               break
            elif len(self.step_history) > 1 and np.linalg.norm(x - self.step_history[-1]) < self.step_tol:
                break
            self.step_history.append(x)

            is_inv = False
            G, is_inv = self.approx_hessian(self.problem.gradient, x)

            # Evaluate Hessian (or inverse) approx
            # G_true = differentiate.hessian(self.problem.objective, x).ddf # TODO Get scipy hess to work with chebyquad
            G_true = finite_diffs(self.problem.gradient, x, dx=1e-7)
            if is_inv:
                G_true = np.linalg.inv(G_true)
            
            # TODO If the Hessian is not symmetric positive definite, this seems to fail...
            if not is_inv:
                # Solve symmetric linear system
                s = linalg.solve(G, -g)
            else:
                # Directly compute step direction
                s = -G @ g
            alpha = self.line_search(x, s)
            self.phi_prime.append(np.dot(self.problem.gradient_at(x), s))

            self.error_history.append(np.linalg.norm(G - G_true, "fro"))
            x = x + alpha * s

        # Return the solution, the and the steps taken, and Hessian errors
        return x

### Classical Newton Implementation

class ClassicalNewton(Optimization):
    """
    Hessian: Approximates the Hessian using finite differences comparing against a given 
    gradient function.

    Line-Search: Uses scipy's optimize_scalar as a shortcut for exact line search.
    """

    def __init__(self, use_exact_ls=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_exact_ls=use_exact_ls
    
    def line_search(self, x, s):
        """
        Uses either the scipy optimize_scalar function for exact line-search, or a method
        implemented from the textbook reference using Armijo-Wolfe conditions.
        """
        
        if self.use_exact_ls:
            return exact_line_search(self.problem.objective, self.problem.gradient, x, s)
        elif self.use_exact_ls == None:
            return 1
        else:
            return arm_wolf_search(self.problem.objective, self.problem.gradient, x, s)


    def approx_hessian(self, g, x):
        """Uses finite differences to estimate the hessian from the gradient."""

        return finite_diffs(g, x), False

### Quasi-Newton Implementations

class QNewton(Optimization):
    """
    Parent class of all Quasi-Newton implementations. Uses line search with Wolfe conditions
    for the line search step, and only expects that subclasses implement the inverse Hessian
    approximation function.

    TODO For now using exact line search as the arm_wolfe currently fails if the step direction
    isn't in a descent direction.
    """

    def line_search(self, x, s):
        """Exact line search."""
        

        return line_search_alt_orsolya(self.problem.objective, self.problem.gradient, x, s)


class GoodBroyden(QNewton):
    """
    Simple Broyden rank-1 update of G and applying Sherman-Morisson's formula ("good Broyden")

    Source: A Class of Methods for Solving Nonlinear Simultaneous Equations
            By C. G. Broyden
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # x from the previous iteration
        self.xp = None
        # f'(x) from the previous iteration
        self.gp = None
        # Previous iteration Hessian (inv) approx
        self.H_inv = None
    
    def reset(self):
        super().reset()
        self.xp = None
        self.gp = None
        self.H_inv = None
    
    def approx_hessian(self, g, x):
        # Update hessian approx
        if not self.xp is None and not self.gp is None:
            v = x - self.xp
            y = g(x) - self.gp
            H = self.H_inv

            denom = np.dot(v, H @ y)
            if abs(denom) > 1e-12:
                # Rank 1 update
                self.H_inv = H + np.outer(v-H@y,v@H) / denom
        else:
            self.H_inv = np.identity(x.shape[0])

        # Store x and g_x for next iteration
        self.xp = x
        self.gp = g(x)

        return self.H_inv, True
    
class BadBroyden(QNewton):
    """
    Simple Broyden rank-1 update of H = G_inv ("bad Broyden")

    Source: A Class of Methods for Solving Nonlinear Simultaneous Equations
        By C. G. Broyden
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # x from the previous iteration
        self.xp = None
        # f'(x) from the previous iteration
        self.gp = None
        # Previous iteration Hessian (inv) approx
        self.H_inv = None
    
    def reset(self):
        super().reset()
        self.xp = None
        self.gp = None
        self.H_inv = None
    
    def approx_hessian(self, g, x):
        # Update hessian approx
        if not self.xp is None and not self.gp is None:
            v = x - self.xp
            y = g(x) - self.gp
            H = self.H_inv

            denom = np.dot(y, y)
            if abs(denom) > 1e-12:
                # Rank 1 update
                self.H_inv = H + np.outer(v-H@y,y) / denom
        else:
            self.H_inv = np.identity(x.shape[0])

        # Store x and g_x for next iteration
        self.xp = x
        self.gp = g(x)

        return self.H_inv, True

class SymmetricBroyden(QNewton):
    """
    Symmetric Broyden rank-2 update, performs Good-Broyden with a symmetrizing
    step.

    Source: A Class of Methods for Solving Nonlinear Simultaneous Equations
            By C. G. Broyden
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # x from the previous iteration
        self.xp = None
        # f'(x) from the previous iteration
        self.gp = None
        # Previous iteration Hessian (inv) approx
        self.H_inv = None
    
    def reset(self):
        super().reset()
        self.xp = None
        self.gp = None
        self.H_inv = None
    
    def approx_hessian(self, g, x):
        # Update hessian approx
        if not self.xp is None and not self.gp is None:
            v = x - self.xp
            y = g(x) - self.gp
            H = self.H_inv

            denom1 = np.dot(y,H@y)
            denom2 = np.dot(v,y)

            if abs(denom1) > 1e-12:
                self.H_inv -= np.outer(H@y,y@H)/denom1
            if abs(denom2) > 1e-12:
                self.H_inv -= np.outer(v,v)/denom2
        else:
            self.H_inv = np.identity(x.shape[0])

        # Store x and g_x for next iteration
        self.xp = x
        self.gp = g(x)

        return self.H_inv, True

class DFP(QNewton):
    """
    Uses the DFP updating formula to update the inv Hessian on each step.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # x from the previous iteration
        self.xp = None
        # f'(x) from the previous iteration
        self.gp = None
        # Previous iteration Hessian (inv) approx
        self.H_inv = None

    def reset(self):
        super().reset()
        self.xp = None
        self.gp = None
        self.H_inv = None
    
    def approx_hessian(self, g, x):
        # Update hessian approx
        if not self.xp is None and not self.gp is None:
            v = x - self.xp
            y = g(x) - self.gp
            H = self.H_inv

            self.H_inv = H - (np.outer(H@y,y@H))/(np.dot(y,H@y)) + (np.outer(v,v))/(np.dot(y,v))
        else:
            self.H_inv = np.identity(x.shape[0])

        # Store x and g_x for next iteration
        self.xp = x
        self.gp = g(x)

        return self.H_inv, True

class BFGS(QNewton):
    """
    Uses the BFGS updating formula to update the inv Hessian on each step.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # x from the previous iteration
        self.xp = None
        # f'(x) from the previous iteration
        self.gp = None
        # Previous iteration Hessian (inv) approx
        self.H_inv = None

    def reset(self):
        super().reset()
        self.xp = None
        self.gp = None
        self.H_inv = None
    
    def approx_hessian(self, g, x):
        """Uses finite differences to estimate the hessian from the gradient."""

        # Update hessian approx
        if not self.xp is None and not self.gp is None:
            v = x - self.xp
            y = g(x) - self.gp
            p = 1 / np.dot(y, v)
            H = self.H_inv
            I = np.identity(x.shape[0])

            self.H_inv = (I-p*np.outer(v,y))@H@(I-p*np.outer(y,v))+p*np.outer(v,v)
        else:
            self.H_inv = np.identity(x.shape[0])

        # Store x and g_x for next iteration
        self.xp = x
        self.gp = g(x)

        return self.H_inv, True

### Inexact Line Search, without Newton's Method

class InexactLineSearch(Optimization):
    def line_search(self, x, s):

        return line_search_alt_orsolya(self.problem.objective, self.problem.gradient, x=x, s=s)

    
    def approx_hessian(self, g, x):
        n = x.shape[0]
        return np.identity(n), True
