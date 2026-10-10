import numpy as np
import scipy.linalg as linalg
from line_search import exact_line_search, arm_wolf_search
from hessian_approx import finite_diffs

class Optimization():
    """
    Optimization problem class.
    """

    def __init__(self, obj_func, obj_grad):
        self.obj_func = obj_func
        self.obj_grad = obj_grad
        self.grad_tol = 1e-6
        self.step_tol = 1e-6
        self.max_steps = 10_000

    def reset(self):
        """
        To be overriden in subclasses that require a reset between solving problems.
        """
        pass

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

        x = np.array(x0)
        x_steps = [x]
        
        for _ in range(self.max_steps):
            g = self.obj_grad(x)

            # Check exit criteria
            if np.linalg.norm(g) < self.grad_tol:
               break
            elif len(x_steps) > 1 and np.linalg.norm(x - x_steps[-1]) < self.step_tol:
                break
            x_steps.append(x)

            is_inv = False
            G, is_inv = self.approx_hessian(self.obj_grad, x)
            # TODO If the Hessian is not symmetric positive definite, this seems to fail...
            if not is_inv:
                # Solve symmetric linear system
                s = linalg.solve(G, -g, assume_a="symmetric")
            else:
                # Directly compute step direction
                s = -G @ g
            alpha = self.line_search(x, s)

            x = x + alpha * s

        # Return the solution, the and the steps taken
        return x, [*x_steps, x]

### Classical Newton Implementation

class ClassicalNewton(Optimization):
    """
    Hessian: Approximates the Hessian using finite differences comparing against a given 
    gradient function.

    Line-Search: Uses scipy's optimize_scalar as a shortcut for exact line search.
    """

    def __init__(self, use_exact_ls=False, *args, **kwargs):
        super(ClassicalNewton, self).__init__(*args, **kwargs)
        self.use_exact_ls=use_exact_ls
    
    def line_search(self, x, s):
        """
        Uses either the scipy optimize_scalar function for exact line-search, or a method
        implemented from the textbook reference using Armijo-Wolfe conditions.
        """
        
        if self.use_exact_ls:
            return exact_line_search(self.obj_func, self.obj_grad, x, s)
        else:
            return arm_wolf_search(self.obj_func, self.obj_grad, x, s)

    def approx_hessian(self, g, x):
        """Uses finite differences to estimate the hessian from the gradient."""

        return finite_diffs(g, x), False

### Quasi-Newton Implementations

class QNewton(Optimization):
    """
    Parent class of all Quasi-Newton implementations. Uses line search with Wolfe conditions
    for the line search step, and only expects that subclasses implement the inverse Hessian
    approximation function.
    """

    def line_search(self, x, s):
        """Exact line search."""
        
        return arm_wolf_search(self.obj_func, self.obj_grad, x, s)

class GoodBroyden(QNewton):
    """
    Simple Broyden rank-1 update of G and applying Sherman-Morisson's formula ("good Broyden")
    """

    def __init__(self, *args, **kwargs):
        super(QNewton, self).__init__(*args, **kwargs)
        # x from the previous iteration
        self.xp = None
        # f'(x) from the previous iteration
        self.gp = None
        # Previous iteration Hessian (inv) approx
        self.H_inv = None
    
    def reset(self):
        self.xp = None
        self.gp = None
        self.H_inv = None
    
    def approx_hessian(self, g, x):
        # Update hessian approx
        if not self.xp is None and not self.gp is None:
            v = x - self.xp
            y = g(x) - self.gp
            H = self.H_inv

            self.H_inv = H + ((v-H@y)/np.dot(v,H@y))@(v@H)
        else:
            self.H_inv = np.identity(x.shape[0])

        # Store x and g_x for next iteration
        self.xp = x
        self.gp = g(x)

        # Symmetrize output
        self.H_inv = (self.H_inv + self.H_inv.transpose())/2
        return self.H_inv, True
    
class BadBroyden(QNewton):
    """
    Simple Broyden rank-1 update of H = G_inv ("bad Broyden")
    """
    
    def __init__(self, *args, **kwargs):
        super(QNewton, self).__init__(*args, **kwargs)
        # x from the previous iteration
        self.xp = None
        # f'(x) from the previous iteration
        self.gp = None
        # Previous iteration Hessian (inv) approx
        self.H_inv = None
    
    def reset(self):
        self.xp = None
        self.gp = None
        self.H_inv = None
    
    def approx_hessian(self, g, x):
        # Update hessian approx
        if not self.xp is None and not self.gp is None:
            v = x - self.xp
            y = g(x) - self.gp
            H = self.H_inv

            self.H_inv = H + ((v-H@y)/np.dot(y,y))@y
        else:
            self.H_inv = np.identity(x.shape[0])

        # Store x and g_x for next iteration
        self.xp = x
        self.gp = g(x)

        # Symmetrize output
        self.H_inv = (self.H_inv + self.H_inv.transpose())/2
        return self.H_inv, True

class SymmetricBroyden(QNewton):
    """
    Symmetric Broyden update, performs Good-Broyden with a symmetrizing
    step.
    """
    
    def __init__(self, *args, **kwargs):
        super(QNewton, self).__init__(*args, **kwargs)
        # x from the previous iteration
        self.xp = None
        # f'(x) from the previous iteration
        self.gp = None
        # Previous iteration Hessian (inv) approx
        self.H_inv = None
    
    def reset(self):
        self.xp = None
        self.gp = None
        self.H_inv = None
    
    def approx_hessian(self, g, x):
        # Update hessian approx
        if not self.xp is None and not self.gp is None:
            v = x - self.xp
            y = g(x) - self.gp
            H = self.H_inv

            self.H_inv = H + (np.outer(y-H@v,v)+np.outer(v,y-H@v))/np.dot(v,v) \
                           - (np.dot(y-H@v,v)/(np.dot(v,v)**2))*np.outer(v,v)
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
        super(QNewton, self).__init__(*args, **kwargs)
        # x from the previous iteration
        self.xp = None
        # f'(x) from the previous iteration
        self.gp = None
        # Previous iteration Hessian (inv) approx
        self.H_inv = None

    def reset(self):
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
        super(QNewton, self).__init__(*args, **kwargs)
        # x from the previous iteration
        self.xp = None
        # f'(x) from the previous iteration
        self.gp = None
        # Previous iteration Hessian (inv) approx
        self.H_inv = None

    def reset(self):
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
        return arm_wolf_search(self.obj_func, self.obj_grad, x=x, s=s)
    
    def approx_hessian(self, g, x):
        n = x.shape[0]
        return np.identity(n), True
