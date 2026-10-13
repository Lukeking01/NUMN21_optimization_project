# NUMN21_optimization_project

## Plotting

Contour plots for the Rosenbrock functions.

Grid plots for the Hessian approximations from Quasi-Newton methods. The current comparison is the approximation from finite differences, but another alternative could be from `scipy.differentiate.hessian`, but that wasn't immediately compatible with the Chebyquad functions.

## Sources

1. Source for the inexact line search with Armijo-Wolfe conditions, and describes the cubic/quadratic interpolation for choosing prospective values of alpha during the line search.
2. Source for the Broyden rank-1 and rank-2 updates.
3. Source for the DFP and BFGS rank-2 updates.

## Images/Outputs

1. Rosenbrock path and Hessian error for:
    - Classical Newton
    - Inexact line search
    - Bad Broyden
    - Good Broyden
    - Symmetric Broyden
    - DFP
    - BFGS
The output contains the number of steps taken. 

#### Classical Newton 
<img width="49%" height="584" alt="Screenshot 2026-10-12 at 22 50 58" src="https://github.com/user-attachments/assets/a68d0f07-50a0-442b-8eb9-2b2d55c87b82" />
<img width="49%" height="482" alt="Screenshot 2026-10-12 at 22 50 40" src="https://github.com/user-attachments/assets/8844e49c-da3a-48a7-a013-8c03068c6a8b" />


#### Inexact line search
<img width="49%" height="586" alt="Screenshot 2026-10-12 at 22 30 45" src="https://github.com/user-attachments/assets/56eea838-674e-4ca1-a3ff-762628db78c1" />
<img width="49%" height="486" alt="Screenshot 2026-10-12 at 22 30 20" src="https://github.com/user-attachments/assets/a0e97745-29fc-46e6-a8d0-eba312925310" />

#### Bad Broyden
<img width="49%" height="584" alt="Screenshot 2026-10-12 at 22 32 13" src="https://github.com/user-attachments/assets/c6c923b4-ce2e-4ff0-ab98-6ebcec81fb8f" />
<img width="49%" height="492" alt="Screenshot 2026-10-12 at 22 31 54" src="https://github.com/user-attachments/assets/435451ae-f612-4eea-84ea-694d30b68db8" />

#### Good Broyden
<img width="49%" height="586" alt="Screenshot 2026-10-12 at 22 48 38" src="https://github.com/user-attachments/assets/36ca03b7-2481-40cc-ba5a-b3b83dd15759" />
<img width="49%" height="482" alt="Screenshot 2026-10-12 at 22 48 14" src="https://github.com/user-attachments/assets/5933dbae-463a-4e19-9510-fc3d97924ec3" />

#### Symmetric Broyden
<img width="49%" height="486" alt="Screenshot 2026-10-12 at 22 45 50" src="https://github.com/user-attachments/assets/5d210095-abdf-423a-942c-a6b853b6bb6d" />
<img width="49%" height="579" alt="Screenshot 2026-10-12 at 22 41 36" src="https://github.com/user-attachments/assets/da1a7304-1fec-438d-8170-7e9b16f6fe06" />

#### DFP
<img width="49%" height="583" alt="Screenshot 2026-10-12 at 22 35 14" src="https://github.com/user-attachments/assets/335b11e9-77ac-4c01-8099-bb55e4e730dc" />
<img width="49%" height="486" alt="Screenshot 2026-10-12 at 22 34 55" src="https://github.com/user-attachments/assets/da70e64a-50cd-4fb0-9d7b-c7952a432168" />

#### BFGS
<img width="49%" height="583" alt="Screenshot 2026-10-12 at 22 36 09" src="https://github.com/user-attachments/assets/a9c30cba-c9f2-405f-927e-40eb8bd1724c" />
<img width="49%" height="483" alt="Screenshot 2026-10-12 at 22 35 50" src="https://github.com/user-attachments/assets/6cbd0fb9-3655-4bc7-90db-bab8e84dc051" />


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

 
