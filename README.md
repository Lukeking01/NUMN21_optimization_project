# NUMN21_optimization_project

## Plotting

Contour plots for the Rosenbrock functions.

Grid plots for the Hessian approximations from Quasi-Newton methods. The current comparison is the approximation from finite differences, but another alternative could be from `scipy.differentiate.hessian`, but that wasn't immediately compatible with the Chebyquad functions.

## Sources

1. Source for the inexact line search with Armijo-Wolfe conditions, and describes the cubic/quadratic interpolation for choosing prospective values of alpha during the line search.
2. Source for the Broyden rank-1 and rank-2 updates.
3. Source for the DFP and BFGS rank-2 updates.

## Images/Outputs

1. Rosenbrock pathing and Hessian error approx graphs. Output could contain the number of steps taken for a given path, or the average for 100 random starting points. This should be for each of the methods we were asked to implement.
    - Classical Newton's Method (exact/inexact line search)
    - Inexact Line Search
    - Broyden's Good/Bad Updates
    - Broyden's Symmetric Updates
    - DFP Updates
    - BFGS Updates
2. Maybe a series of images showing an interesting example of line search? Could include graphs showing the process when just bisecting the interval each time, and another with the cubic interpolation logic.

## Assigment Tasks

### Task 5 
Testing the performance of the Rosenbrock's function and demonstrating its behaviour with a contour plot.

### Task 7
Testing the inexact line search method on the Rosenbrock's function with the parameters given on page 37 of the book.

### Task 11
Computing the chebyquad function provided for the points $n=4,8,11$ and comparing it to the built in BFGS function.

### Task 12
Showing the quality of approximation of the inverse of the Hessian matrix with a growing number of iterations.

## TODO/IMPROVEMENTS

- Make sure line search and all optimization methods do not crash often or at all on random starting locations.
- Add full documentation to functions, and ensure code is well organized and parameterized.
- Add tests?
- Want to add CG and LBFGS method and comparison plot of their convergence (if time permits) - Orsi

## Contributions
Each team member has done the code individually and then we combined them into one coherent document.

 
