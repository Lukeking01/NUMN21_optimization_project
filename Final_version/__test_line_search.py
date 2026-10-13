from Rosenbrock_problem import rosenbrock, rosenbrock_gradient
from optimization import *
from plot import *

def run_rosenbrock(method, x_vals, with_plotting=True):
    step_count = 0
    ascent_dir_count = 0
    
    for x in x_vals:
        sol = method.solve(x)
        steps = method.step_history
        H_errors = method.error_history

        step_count += (len(steps)-1)
        ascent_dir_count += method.ascent_dir_count
        print(f"Found solution {sol} in {len(steps)-1} steps")
        if with_plotting:
            draw_contour(
                rosenbrock_fn, steps=steps, print_labels=False, n_levels=14
            )

            draw_error_grid(H_errors)
    print(f"AVERAGE STEPS TAKEN: {step_count / x_vals.shape[0]}")
    print(f"AVERAGE ASCENT DIRS: {ascent_dir_count / x_vals.shape[0]}")

# Setup problems
rose_pr = Problem(objective=rosenbrock, gradient=rosenbrock_gradient)
rose_pr_nograd = Problem(objective=rosenbrock, gradient=None)

# Choose method
# method = ClassicalNewton(problem=rose_pr, use_exact_ls=False)
# method = InexactLineSearch(problem=rose_pr)
# method = GoodBroyden(problem=rose_pr) # 18.05 / 61.95
# method = SymmetricBroyden(problem=rose_pr) # 15.26 / 16.25
# method = BadBroyden(problem=rose_pr) # TODO Not working, is this expected?
# method = DFP(problem=rose_pr) # 16.02 / 17.02
method = BFGS(problem=rose_pr) # 0 / 14.66

if __name__ == "__main__":
    FAILED_POINTS = np.array([
        [-1.18315709,  1.40683587],
        [0.95000535, 0.90914311],
        [0.4431342 , 0.21549486],
    ])

    # points = FAILED_POINTS
    points = np.random.random((100, 2)) * 4 - 1
    run_rosenbrock(method, points, with_plotting=False)
