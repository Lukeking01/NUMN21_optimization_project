import numpy as np
import scipy.optimize as so
from optimization import *
from chebyquad_problem import chebyquad, gradchebyquad
from plot import draw_contour, draw_error_grid
from constants import *
import matplotlib.pyplot as plt

def rosenbrock_fn(x):
    return 100 * ((x[1] - x[0]**2)**2) + (1 - x[0])**2
def rosenbrock_grad(x):
    grad1 = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    grad2 = 200 * (x[1] - x[0]**2)
    
    return np.array([grad1, grad2])

def run_rosenbrock(method, x_vals):
    for x in x_vals:
        method.reset()
        sol, steps, H_errors = method.solve(x)
        print(f"Found solution {sol} in {len(steps)-1} steps")
        draw_contour(
            rosenbrock_fn, steps=steps, print_labels=False, n_levels=14
        )

        draw_error_grid(H_errors)

def run_chebyquad(method, size=8):
    x = np.linspace(0,1,size)
    method.reset()
    
    sol, steps, H_errors = method.solve(x)
    print(f"Classical Newton Method -- {size=}:")
    print(f"\tCurrent function value: {chebyquad(sol)}")
    print(f"\tIterations: {len(steps)-1}")
    print(f"\tFunction evaluations: TODO!")
    print(f"\tGradient evaluations: TODO!")

    draw_error_grid(H_errors)

    xmin = so.fmin_bfgs(chebyquad,x,gradchebyquad)
    print("FMIN_BFGS\n\n")


if __name__ == "__main__":
    # method = ClassicalNewton(obj_func=rosenbrock_fn, obj_grad=rosenbrock_grad, use_exact_ls=True)
    # method = ClassicalNewton(obj_func=rosenbrock_fn, obj_grad=rosenbrock_grad, use_exact_ls=False)
    # method = InexactLineSearch(obj_func=rosenbrock_fn, obj_grad=rosenbrock_grad)
    # method = GoodBroyden(obj_func=rosenbrock_fn, obj_grad=rosenbrock_grad)
    # method = SymmetricBroyden(obj_func=rosenbrock_fn, obj_grad=rosenbrock_grad)
    # method = BadBroyden(obj_func=rosenbrock_fn, obj_grad=rosenbrock_grad)
    # method = DFP(obj_func=rosenbrock_fn, obj_grad=rosenbrock_grad)
    method = BFGS(obj_func=rosenbrock_fn, obj_grad=rosenbrock_grad)
    
    x_vals = np.random.random((8, 2)) * 8 - 6
    # x_vals = FAILED_POINTS

    run_rosenbrock(method, x_vals)
    
    # method = ClassicalNewton(obj_func=chebyquad, obj_grad=gradchebyquad, use_exact_ls=True)
    # method = ClassicalNewton(obj_func=chebyquad, obj_grad=gradchebyquad, use_exact_ls=False)
    # method = InexactLineSearch(obj_func=chebyquad, obj_grad=gradchebyquad)
    # method = GoodBroyden(obj_func=chebyquad, obj_grad=gradchebyquad)
    # method = SymmetricBroyden(obj_func=chebyquad, obj_grad=gradchebyquad)
    # method = BadBroyden(obj_func=chebyquad, obj_grad=gradchebyquad)
    # method = DFP(obj_func=chebyquad, obj_grad=gradchebyquad)
    method = BFGS(obj_func=chebyquad, obj_grad=gradchebyquad)

    run_chebyquad(method, 4)
    run_chebyquad(method, 8)
    run_chebyquad(method, 11)
    run_chebyquad(method, 14)
