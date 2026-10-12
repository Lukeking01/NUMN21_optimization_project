import numpy as np
import scipy.optimize as so

### HELPER FUNCTIONS ###

def _line_search_fns(obj_func, obj_grad, x, s):
    """
    Get the line search functions phi and phi_prime where:
    | phi(a) = x + as | with "x" and "s" given.

    Returns phi, phi_prime
    """
    def phi(a):
        return obj_func(x + s * a)
    def phi_prime(a):
        return np.dot(obj_grad(x + s * a), s)      

    return phi, phi_prime

### LINE SEARCH FUNCTIONS ###

def exact_line_search(obj_func, obj_grad, x, s):
    """
    Exact line search using scipy's minimize scalar function on the function
    | phi(a) = x + as | with "x" and "s" given.
    """

    phi, phi_prime = _line_search_fns(obj_func, obj_grad, x, s)

    rho = 0.01
    mu = -phi(0) / (rho * phi_prime(0))
    res = so.minimize_scalar(phi, bracket=[0, mu])
    return res.x

def arm_wolf_search(obj_func, obj_grad, x, s):
    """
    Line search based on Armijo-Wolfe conditions.
    
    :param obj_func: Objective function
    :param obj_grad: Objective function gradient
    :param x: Current location
    :param s: Current step direction

    :returns float: The step size determined by the Armijo-Wolfe conditions.
    """
    phi, phi_prime = _line_search_fns(obj_func, obj_grad, x, s)

    if phi_prime(0) > 0:
        raise ValueError(f"{s=} is not a valid step direction for {x=}!")

    # Constants
    max_steps = 100
    lower_bound = 0
    sigma, rho = 0.1, 0.01
    tau1, tau2, tau3 = 9, 0.1, 0.5

    # Pick new value in interval by minimizing a cubic polynomial
    def pick_value_cubic(a, b):
        f0 = phi(a)
        f1 = phi(b)
        fp0 = (b-a)*phi_prime(a)
        fp1 = (b-a)*phi_prime(b)

        eta = 3*(f1-f0)-2*fp0-fp1
        xi = fp0+fp1-2*(f1-f0)

        # Solve for minima of the cubic

        # Boundary points
        cub0 = f0
        cub1 = f0 + fp0 + eta + xi
        
        # Stationary points (quadratic roots)
        st_pts = []
        if abs(xi) > 1e-5:
            root_sq = 4*(eta**2) - 12*xi*fp0
            if root_sq >= 0:
                lroot = (-2*eta - np.sqrt(root_sq)) / (6*xi)
                rroot = (-2*eta + np.sqrt(root_sq)) / (6*xi)

                st_pts.append(lroot)
                st_pts.append(rroot)
        elif abs(eta) > 1e-5:
            root = -fp0 / (2*eta)
            st_pts.append(root)

        min_z = 0 if cub0 < cub1 else 1
        for stpt in st_pts:
            if stpt < 0 or stpt > 1:
                continue
            
            if phi(a + stpt*(b - a)) < phi(a + min_z*(b - a)):
                min_z = stpt

        # Convert back to alpha in range a, b
        return a + min_z*(b - a)

    # Solve bracket
    def solve_bracket(a, b):
        for _ in range(1, max_steps):
            alpha = pick_value_cubic(a + tau2*(b-a), b - tau3*(b-a))
            if phi(alpha) > phi(0) + rho * alpha * phi_prime(0) or phi(alpha) >= phi(a):
                b = alpha
            else:
                if abs(phi_prime(alpha)) <= -sigma * phi_prime(0):
                    return alpha
                a_prev = a
                a = alpha
                if (b - a_prev) * phi_prime(alpha) >= 0:
                    b = a_prev
        return alpha

    # Bracketing interval
    mu = (lower_bound - phi(0)) / (rho * phi_prime(0))
    a, b = 0, mu

    # Pick an initial guess
    alpha_prev = 0
    alpha = a + mu / 4
    for _ in range(1, max_steps):
        if phi(alpha) < lower_bound:
            return alpha
        if phi(alpha) > phi(0) + alpha * rho * phi_prime(0) or phi(alpha) >= phi(alpha_prev):
            return solve_bracket(alpha_prev, alpha)
        if abs(phi_prime(alpha)) <= -sigma * phi_prime(0):
            return alpha
        if phi_prime(alpha) >= 0:
            return solve_bracket(alpha, alpha_prev)
        if mu <= 2 * alpha - alpha_prev:
            tmp = alpha
            alpha = mu
            alpha_prev = tmp
        else:
            tmp = alpha
            alpha = pick_value_cubic(2*alpha - alpha_prev, min(mu, alpha + tau1*(alpha-alpha_prev)))
            alpha_prev = tmp
    return alpha