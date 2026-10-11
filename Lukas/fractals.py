import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import optimization_problem as op
from optimization_problem import BFGS, NewtonMethod

def double_well(x):
    return (x[0]**2 - 1)**2 + x[1]**2

def double_well_grad(x):
    return np.array([4*x[0]*(x[0]**2 - 1), 2*x[1]])
# minima approximately at (±1, 0)

def three_hump_camel(x):
    x1, x2 = x
    return 2*x1**2 - 1.05*x1**4 + x1**6/6 + x1*x2 + x2**2

def three_hump_camel_grad(x):
    x1, x2 = x
    df1 = 4*x1 - 4.2*x1**3 + x1**5 + x2
    df2 = x1 + 2*x2
    return np.array([df1, df2])
# minima near (0,0), (±1.75, ∓0.87)

def himmelblau(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

def himmelblau_grad(x):
    x1, x2 = x
    df1 = 4*x1*(x1**2 + x2 - 11) + 2*(x1 + x2**2 - 7)
    df2 = 2*(x1**2 + x2 - 11) + 4*x2*(x1 + x2**2 - 7)
    return np.array([df1, df2])

def f(z):
    x = z[0]+1j*z[1]
    return x**4 - 2*x**2 + 1

def grad(z):
    x = z[0]+1j*z[1]
    return 4*x**3 - 4*x
# The four known minima (for reference / labelling)
true_minima = np.array([
    [ 3.0,       2.0      ],
    [-2.805118,  3.131312 ],
    [-3.779310, -3.283186 ],
    [ 3.584428, -1.848126 ]
])
true_minima = np.array([1.0,
                -1.0])
problem = op.OptimizationProblem(f, grad)

# ------------------------------------------------------------------
# Grid
# ------------------------------------------------------------------
num = 4         # 250×250 = 62 500 runs – good balance of speed/detail
x = np.linspace(-6, 6, num)
y = np.linspace(-6, 6, num)
X, Y = np.meshgrid(x, y)

Z = np.full((num, num), -1, dtype=int)   # colour = index of attractor

tol = 0.15         # how close a solution must be to a known minimum

bfgs = NewtonMethod(problem, max_iterations=100, tolerance=1e-8, hessian_step=1e-1)

for i in range(num):
    for j in range(num):
        x0 = (X[i, j],Y[i,j])
        sol = bfgs.minimize(x0)
        # find the closest known minimum
        dists = np.abs(true_minima - sol[-1])
        k = np.argmin(dists)
        print(sol)
        if dists[k] < tol:
            Z[i, j] = k
        else:
            Z[i, j] = 4          # “did not converge to any known min”
    print(i)

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#999999']  # 4 minima + grey
cmap = ListedColormap(colors)

plt.figure(figsize=(9, 8))
im = plt.imshow(Z, extent=[x.min(), x.max(), y.min(), y.max()],
                origin='lower', vmin=-0.5, vmax=4.5,
                interpolation='nearest', aspect='equal')

# mark the true minima
# for k, mx in enumerate(true_minima):
#     plt.plot(mx, 'k*', markersize=14, markeredgecolor='white', markeredgewidth=0.8)

plt.tight_layout()
plt.show()