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

def good_broyden_scott(H, s, y):
    """
    :param H: Previous Hess inv approx
    :param s: Delta between previous location and current location.
    :param y: Delta between the current grad and the previous grad.
    """
    Hy = H@y
    denom = np.dot(s, Hy)
    if abs(denom) > 1e-12:
        # Rank 1 update
        H = H + np.outer(s-Hy,s@H)/denom
    return H
def good_broyden_linn(H, s, y):
    """Same as Scott's implementation."""
    Hg = H@y
    denom = s@Hg
    if abs(denom) < 1e-12:
        return H
    return H + np.outer(s-Hg, s@H)/denom
def good_broyden_orsolya(H, s, y):
    """Same as Scott's implementation."""
    u = s-H@y
    denom = np.dot(s, H@y)
    return H+np.outer(u, s@H)/denom
# def good_broyden_lukas(H, s, y):
#     if np.dot(s, y) <= 1e-14:
#         return H
#     Gs = self.G @ s
#     self.G = self.G + np.outer(y - Gs, s) / np.dot(s, s)

#     u = y - Gs
#     v = s / np.dot(s, s)
#     den = 1.0 + v @ self.H @ u
#     if abs(den) > 1e-14:
#         self.H = self.H - np.outer(self.H @ u, v @ self.H) / den
#     else:
#         self.H = np.linalg.pinv(self.G)

def bad_broyden_scott(H, s, y):
    """
    :param H: Previous Hess inv approx
    :param s: Delta between previous location and current location.
    :param y: Delta between the current grad and the previous grad.
    """
    denom = np.dot(y, y)
    if abs(denom) > 1e-12:
        # Rank 1 update
        H = H + np.outer(s-H@y,y) / denom
    return H
def bad_broyden_linn(H, s, y):
    pass
def bad_broyden_orsolya(H, s, y):
    pass
def bad_broyden_lukas(H, s, y):
    pass

def sym_broyden_scott(H, s, y):
    """
    :param H: Previous Hess inv approx
    :param s: Delta between previous location and current location.
    :param y: Delta between the current grad and the previous grad.
    """
    denom1 = np.dot(y,H@y)
    denom2 = np.dot(s,y)

    if abs(denom1) > 1e-12:
        H = H - np.outer(H@y,y@H)/denom1
    if abs(denom2) > 1e-12:
        H = H - np.outer(s,s)/denom2
    return H
def sym_broyden_linn(H, s, y):
    pass
def sym_broyden_orsolya(H, s, y):
    pass
def sym_broyden_lukas(H, s, y):
    pass

def DFP_scott(H, s, y):
    """
    :param H: Previous Hess inv approx
    :param s: Delta between previous location and current location.
    :param y: Delta between the current grad and the previous grad.
    """
    denom1 = np.dot(y,H@y)
    denom2 = np.dot(y,s)

    if abs(denom1) > 1e-12:
        H = H - np.outer(H@y,y@H)/denom1
    if abs(denom2) > 1e-12:
        H = H - np.outer(s,s)/denom2
    return H
def DFP_linn(H, s, y):
    pass
def DFP_orsolya(H, s, y):
    pass
def DFP_lukas(H, s, y):
    pass

def BFGS_scott(H, s, y):
    """
    :param H: Previous Hess inv approx
    :param s: Delta between previous location and current location.
    :param y: Delta between the current grad and the previous grad.
    """
    sy = np.dot(y, s)
    if abs(sy) <= 1e-12:
        return H
    p = 1 / sy
    I = np.identity(s.shape[0])

    H = (I-p*np.outer(s,y))@H@(I-p*np.outer(y,s))+p*np.outer(s,s)
    return H
def BFGS_linn(H, s, y):
    pass
def BFGS_orsolya(H, s, y):
    pass
def BFGS_lukas(H, s, y):
    pass
