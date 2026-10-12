import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import optimization as op
from optimization import ClassicalNewton, BFGS

# ------------------------------------------------------------------
# Complex Newton as a real 2-D optimization problem
# Minimize |z³ - 1|²  where z = x + i y
# ------------------------------------------------------------------
def f(xy):
    x, y = xy
    # z³ - 1 = (x³ - 3xy² - 1) + i(3x²y - y³)
    re = x**3 - 3*x*y**2 - 1
    im = 3*x**2*y - y**3
    return re**2 + im**2          # |z³ - 1|²

def grad(xy):
    # analytic gradient of |z³ - 1|²
    x, y = xy
    re = x**3 - 3*x*y**2 - 1
    im = 3*x**2*y - y**3
    # ∂re/∂x = 3x² - 3y²,  ∂re/∂y = -6xy
    # ∂im/∂x = 6xy,        ∂im/∂y = 3x² - 3y²
    dfdx = 2*re*(3*x**2 - 3*y**2) + 2*im*(6*x*y)
    dfdy = 2*re*(-6*x*y)          + 2*im*(3*x**2 - 3*y**2)
    return np.array([dfdx, dfdy])

problem = op.Optimization(f, grad)

# The three roots of z³ = 1
roots = np.array([
    [1.0,  0.0],
    [-0.5,  0.86602540378],
    [-0.5, -0.86602540378]
])

def run_one_Newton(xy):
    """Run NewtonMethod and return which root it found + iterations"""
    solver = ClassicalNewton(problem, max_steps=100)
    sol = solver.minimize(np.asarray(xy, dtype=float))
    
    # which root
    dists = np.linalg.norm(roots - sol, axis=1)
    k = np.argmin(dists)
    if dists[k] < 0.1:
        return k, len(solver.history)-1
    else:
        return 3, len(solver.history)-1   # did not converge
    
def run_one_BFGS(xy):
    """Run BFGS and return which root it found + iterations"""
    solver = BFGS(problem, max_steps=100)
    sol = solver.minimize(np.asarray(xy, dtype=float))
    
    # which root
    dists = np.linalg.norm(roots - sol, axis=1)
    k = np.argmin(dists)
    if dists[k] < 0.1:
        return k, len(solver.history)-1
    else:
        return 3, len(solver.history)-1   # did not converge

# ------------------------------------------------------------------
# parallel computation
# ------------------------------------------------------------------

def fractal(resolution, run_one, path="imgs/newton_fractal.png"):
    num = resolution          # increase to 800–1000 for higher quality
    re = np.linspace(-1.5, 1.5, num)
    im = np.linspace(-1.5, 1.5, num)
    X, Y = np.meshgrid(re, im)
    points = np.column_stack([X.ravel(), Y.ravel()])

    print("Computing Newton fractal")
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(run_one, points, chunksize=32))

    colors = np.array([r[0] for r in results]).reshape(num, num)
    iters  = np.array([r[1] for r in results]).reshape(num, num)

    # ------------------------------------------------------------------
    # Plot two versions
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Coloured by which root
    im0 = axes[0].imshow(colors,
                            origin='lower', interpolation='nearest')
    axes[0].set_title("coloured by root")
    axes[0].set_xlabel("Re(z)")
    axes[0].set_ylabel("Im(z)")

    # 2. Coloured by iteration count 
    im1 = axes[1].imshow(iters, extent=[-1.5, 1.5, -1.5, 1.5],
                            origin='lower', cmap='magma', interpolation='nearest')
    axes[1].set_title("coloured by iterations")
    axes[1].set_xlabel("Re(z)")
    axes[1].set_ylabel("Im(z)")
    plt.colorbar(im1, ax=axes[1], label="Iterations")

    plt.tight_layout()
    # plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.show()
    
if __name__ == "__main__":
    fractal(8, run_one_Newton)