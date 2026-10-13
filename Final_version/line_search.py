import numpy as np
import scipy.optimize as so

### HELPER FUNCTIONS ###

def _line_search_fns(problem, x, s):
    """
    Get the line search functions phi and phi_prime where:
    | phi(a) = x + as | with "x" and "s" given.

    Returns phi, phi_prime
    """
    def phi(a):
        return problem.objective(x + s * a)
    def phi_prime(a):
        return np.dot(problem.gradient(x + s * a), s)

    return phi, phi_prime

### LINE SEARCH FUNCTIONS ###

def exact_line_search(problem, x, s):
    """
    Exact line search using scipy's minimize scalar function on the function
    | phi(a) = x + as | with "x" and "s" given.
    """

    phi, phi_prime = _line_search_fns(problem, x, s)

    rho = 0.01
    mu = -phi(0) / (rho * phi_prime(0))
    res = so.minimize_scalar(phi, bracket=[0, mu])
    return res.x

def arm_wolf_search(problem, x, s, alpha0 = 1.0):
    """
    Line search based on Armijo-Wolfe conditions.
    
    :param obj_func: Objective function
    :param obj_grad: Objective function gradient
    :param x: Current location
    :param s: Current step direction

    :returns float: The step size determined by the Armijo-Wolfe conditions.
    """
    phi, phi_prime = _line_search_fns(problem, x, s)

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
        new_alpha = a + min_z*(b - a)
        if (new_alpha > a*1.02 and new_alpha > b*1.02) or (new_alpha < a*0.98 and new_alpha < b*0.98):
            raise ValueError("Cubic interval yielded invalid value outside of range.")
        return new_alpha

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
    alpha = alpha0
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

def line_search_alt_orsolya(func, gradient, x, s, rho=1e-4, sigma=0.9, max_alpha=10.0, max_iter=1000):
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
        f_j = func(x_j)
        f_low = func(x+alpha_low*s)

        if (f_j>f0+rho*alpha_j*dphi0) or (f_j>=f_low):
            alpha_high = alpha_j
        else:
            g_j = gradient(x_j)
            dphi_j = np.dot(g_j, s)

            if np.abs(dphi_j)<=-sigma*dphi0:
                return alpha_j

            if dphi_j*(alpha_high-alpha_low)>=0:
                alpha_high = alpha_low

            alpha_low = alpha_j

    return alpha_j
