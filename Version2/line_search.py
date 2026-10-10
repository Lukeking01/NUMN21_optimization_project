import numpy as np


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
