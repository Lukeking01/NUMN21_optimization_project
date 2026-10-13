from Rosenbrock_problem import rosenbrock, rosenbrock_gradient
from chebyquad_problem import chebyquad, gradchebyquad
from optimization import *
from __test_line_search import run_rosenbrock

rose_pr = Problem(rosenbrock, rosenbrock_gradient)
cheb_pr = Problem(chebyquad, gradchebyquad)

method = GoodBroyden(problem=rose_pr)
# method = GoodBroyden(cheb_pr)

if __name__ == "__main__":
    FAILED_POINTS = np.array([
        [-1.18315709,  1.40683587],
        [0.95000535, 0.90914311],
        [0.4431342 , 0.21549486],
    ])
    # points = FAILED_POINTS
    points = np.random.random((20, 2)) * 3.5 - 1
    run_rosenbrock(method, points, True)
