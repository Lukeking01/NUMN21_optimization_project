from optimization import *
from plot import *

def rosenbrock_fn(x):
    return 100 * ((x[1] - x[0]**2)**2) + (1 - x[0])**2
def rosenbrock_grad(x):
    grad1 = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    grad2 = 200 * (x[1] - x[0]**2)
    
    return np.array([grad1, grad2])

def run_rosenbrock(method, x_vals):
    for x in x_vals:
        sol = method.solve(x)
        steps = method.step_history
        H_errors = method.error_history
        
        print(f"Found solution {sol} in {len(steps)-1} steps")
        draw_contour(
            rosenbrock_fn, steps=steps, print_labels=False, n_levels=14
        )

        draw_error_grid(H_errors)

method = ClassicalNewton(obj_func=rosenbrock_fn, obj_grad=rosenbrock_grad, use_exact_ls=False)
# method = ClassicalNewton(obj_func=rosenbrock_fn, obj_grad=rosenbrock_grad, use_exact_ls=False)
# method = InexactLineSearch(obj_func=rosenbrock_fn, obj_grad=rosenbrock_grad)
# method = GoodBroyden(obj_func=rosenbrock_fn, obj_grad=rosenbrock_grad)
# method = SymmetricBroyden(obj_func=rosenbrock_fn, obj_grad=rosenbrock_grad)
# method = BadBroyden(obj_func=rosenbrock_fn, obj_grad=rosenbrock_grad)
# method = DFP(obj_func=rosenbrock_fn, obj_grad=rosenbrock_grad)
# method = BFGS(obj_func=rosenbrock_fn, obj_grad=rosenbrock_grad)

FAILED_POINTS = [
    [-3.0644475476131317, 9.742501866982106],
    [-1.1263393130418025, 1.4502462268373029],
    [-1.18315709,  1.40683587],
]
run_rosenbrock(method, FAILED_POINTS)