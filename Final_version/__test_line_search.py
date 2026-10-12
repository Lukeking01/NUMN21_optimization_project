from Rosenbrock_problem import rosenbrock, rosenbrock_gradient
from optimization import *
from plot import *


def run_rosenbrock(method, x_vals):
    for x in x_vals:
        sol = method.solve(x)
        steps = method.step_history
        H_errors = method.error_history
        phi0 = np.array(method.phi_prime)

        print("--- DEBUG phi'(0) ---")
        print(f"{np.min(phi0)} | {np.max(phi0)}")
        
        print(f"Found solution {sol} in {len(steps)-1} steps")
        draw_contour(
            rosenbrock_fn, steps=steps, print_labels=False, n_levels=14
        )

        draw_error_grid(H_errors)

# Setup problems
rose_pr = Problem(objective=rosenbrock, gradient=rosenbrock_gradient)
rose_pr_nograd = Problem(objective=rosenbrock, gradient=None)

# Choose method
method = ClassicalNewton(problem=rose_pr, use_exact_ls=False)
# method = InexactLineSearch(problem=rose_pr)
# method = GoodBroyden(problem=rose_pr)
# method = SymmetricBroyden(problem=rose_pr)
# method = BadBroyden(problem=rose_pr)
# method = DFP(problem=rose_pr)
# method = BFGS(problem=rose_pr)

FAILED_POINTS = [
    [-3.0644475476131317, 9.742501866982106],
    [-1.1263393130418025, 1.4502462268373029],
    [-1.18315709,  1.40683587],
]
run_rosenbrock(method, FAILED_POINTS)