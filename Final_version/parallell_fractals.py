import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import optimization as op
from optimization import ClassicalNewton, BFGS, DFP

# ------------------------------------------------------------------
# Complex Newton as a real 2-D optimization problem
# Minimize |z³ - 1|²  where z = x + i y
# ------------------------------------------------------------------
def f(xy):
    '''function |z³ - 1|², where z = x + i y '''
    x, y = xy
    # z³ - 1 = (x³ - 3xy² - 1) + i(3x²y - y³)
    re = x**3 - 3*x*y**2 - 1
    im = 3*x**2*y - y**3
    return re**2 + im**2          # |z³ - 1|²

def grad(xy):
    '''analytic gradient of |z³ - 1|²'''
    x, y = xy
    re = x**3 - 3*x*y**2 - 1
    im = 3*x**2*y - y**3

    dfdx = 2*re*(3*x**2 - 3*y**2) + 2*im*(6*x*y)
    dfdy = 2*re*(-6*x*y)          + 2*im*(3*x**2 - 3*y**2)
    return np.array([dfdx, dfdy])

problem = op.Problem(f, grad)

# The three roots of z³ = 1
roots = np.array([
    [1.0,  0.0],
    [-0.5,  0.86602540378],
    [-0.5, -0.86602540378]
])


def run_one_Newton(xy):
    """Run NewtonMethod and return which root it found + iterations"""
    solver = ClassicalNewton(None,problem)
    sol = solver.solve(np.asarray(xy, dtype=float))
    
    # which root
    dists = np.linalg.norm(roots - sol, axis=1)
    k = np.argmin(dists)

    if dists[k] < 0.1:
        return k, len(solver.step_history)-1
    else:
        return 3, len(solver.step_history)-1   # did not converge
    
def run_one_BFGS(xy):
    """Run BFGS and return which root it found + iterations"""
    solver = BFGS(problem)
    sol = solver.solve(np.asarray(xy, dtype=float))
    
    # which root
    dists = np.linalg.norm(roots - sol, axis=1)
    k = np.argmin(dists)
    if dists[k] < 0.1:
        return k, len(solver.step_history)-1
    else:
        return 3, len(solver.step_history)-1   # did not converge

def run_one_DFP(xy):
    """Run BFGS and return which root it found + iterations"""
    solver = DFP(problem)
    sol = solver.solve(np.asarray(xy, dtype=float))
    
    # which root
    dists = np.linalg.norm(roots - sol, axis=1)
    k = np.argmin(dists)
    if dists[k] < 0.1:
        return k, len(solver.step_history)-1
    else:
        return 3, len(solver.step_history)-1   # did not converge
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
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), dpi=300)

    # 1. Coloured by which root
    im0 = axes[0].imshow(colors,
                            origin='lower', interpolation='nearest')
    axes[0].set_title("coloured by root")
    axes[0].set_xticks([])
    axes[0].set_yticks([])


    # 2. Coloured by iteration count 
    im1 = axes[1].imshow(iters,
                            origin='lower', cmap='magma', interpolation='nearest', vmax=40)
    axes[1].set_title("coloured by iterations")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    #plt.colorbar(im1, ax=axes[1], label="Iterations")

    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=300, pad_inches=0)
    plt.show()
    
if __name__ == "__main__":
    res = 1800
    # fractal(res, run_one_Newton, path=None)#"imgs/newton_fractal.png")
    # fractal(res, run_one_BFGS, path=None)#"imgs/BFGS_fractal.png")
    fractal(res, run_one_DFP, path="imgs/DFP_fractal.png")


