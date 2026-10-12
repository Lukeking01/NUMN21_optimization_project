import numpy as np

def finite_diffs(g, x, dx=1e-4):
    """
    Approximate the hessian at a given point by measuring the change of derivatives
    with central finite differences.

    :param g: Gradient function
    :param x: Point of interest
    :returns ndarray: Approximation of Hessian at point x of g.
    """

    g_x = g(x)
    n = g_x.shape[0]

    G = np.zeros((n, n))
    for i in range(0, n):
        e_i = np.zeros(n)
        e_i[i] = 1.0

        # Central difference formula
        g_forw = g(x + dx * e_i).flatten()
        g_back = g(x - dx * e_i).flatten()
        
        G[:,i] = (g_forw - g_back) / (2 * dx)

    # Symmetrize
    return (G + G.transpose()) / 2
    