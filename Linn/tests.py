import numpy as np
import matplotlib.pyplot as plt
from optimization import optimization, NewtonMethod, line_search_wolfe, BadBroyden, GoodBroyden, symmetricBroyden, DFP, BFGS
from numpy import dot, linspace
from scipy.optimize import fmin_bfgs
from chebyquad_problem import chebyquad, gradchebyquad

def rosenbrock(x):
    return 100.0* (x[1]-x[0]**2) **2 + (1.0-x[0])**2

def rosen_grad(x):
    grad = np.array([
        -400.0*x[0]* (x[1] - x[0] **2) -2.0*(1.0-x[0]),
        200.0* (x[1]- x[0]**2),
    ])
    return grad

def rosen_hess(x):
    hess= np.array([[1200.0*x[0]**2 - 400.0*x[1]+2.0, -400.0*x[0]],
                    [-400.0*x[0], 200.0],
                    ])
    return hess

def task5():
    print('test with Rosenbrock function')

    prob = optimization(rosenbrock, rosen_grad, rosen_hess)
    solver = NewtonMethod(prob, x0=np.array([-1.2, 1.0]),
                          tol=1e-10, max_iter=50, exact_line_search=True)
    
    xstar = solver.solve()

    for k in range(len(solver.x_history)):
        xk = solver.x_history[k]
        fk = solver.f_history[k]
        gk = np.linalg.norm(solver.g_history[k])
        ak = solver.alpha_history[k-1] if k>0 else None

    fig,ax = plt.subplots(figsize=(7,6))
    X,Y= np.meshgrid(np.linspace(-2.0,2.0,400), np.linspace(-1.0,3.0,400))
    Z = 100.0 * (Y-X **2) **2 + (1.0-X)**2
    ax.contour(X,Y,Z, levels=np.logspace(-1,3,20))

    hist=np.array(solver.x_history)
    ax.plot(hist[:,0], hist[:,1], 'o-', color='C3', label='Newton iterates')
    ax.plot(hist[0,0], hist[0,1], 's', color='k', markersize=10, label=r'$x^{(0)}$')
    ax.plot(1.0,1.0,'r*', markersize=15, label=r'$x^*=(1,1)$')

    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_title('Newton on Rosenbrock (task5)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def task7():
    prob = optimization(rosenbrock, rosen_grad)
    rho, sigma = 0.01, 0.1
    for x0 in [np.array([-1.2, 1.0]), np.array([0.0,0.0]), np.array([2.0,2.0])]:
        s = -rosen_grad(x0)
        a = line_search_wolfe(prob, x0,s, rho=rho, sigma=sigma)
        d0 = rosen_grad(x0) @ s
        print(x0,a,rosenbrock(x0+a*s)<=rosenbrock(x0)+ rho*a*d0, #armijo
              abs(rosen_grad(x0 +a*s)@s) <= -sigma*d0) #curvature


# to check task8
def task8():
    prob = optimization(rosenbrock, rosen_grad, rosen_hess)
    x0 = np.array([-1.2, 1.0])

    exact = NewtonMethod(prob, x0, tol=1e-10, max_iter=50, exact_line_search=True)
    exact.solve()

    inexact = NewtonMethod(prob, x0, tol=1e-10, max_iter=50, exact_line_search=False)
    inexact.solve()

    print("exact", len(exact.alpha_history), "iterations")
    print("wolfe", len(inexact.alpha_history), "iterations")


def task9():
    prob = optimization(rosenbrock, rosen_grad)
    for C in [GoodBroyden, BadBroyden, symmetricBroyden, DFP, BFGS]:
        m = C(prob, [-1.2, 1.0], tol=1e-8, max_iter=500, line_search='wolfe')
        x = m.solve()
        print(C.__name__, len(m.alpha_history),x)

def task11():
    prob = optimization(chebyquad, gradchebyquad)
    for n in [4,8,11]:
        x0 = np.linspace(0,1,n)
        m = BFGS(prob, x0, tol=1e-8, max_iter=1000,line_search='wolfe')
        x = m.solve()
        xs = fmin_bfgs(chebyquad, x0, fprime=gradchebyquad, gtol=1e-8, disp=False)

        print(f'n={n}: interations={len(m.alpha_history)}')
        print(f'mine:', chebyquad(x), np.sort(x))
        print('scipy:', chebyquad(xs), np.sort(xs))



#task12
def bfgs_hessian_quality(prob, x0, label):
    m = BFGS(prob, x0, tol=1e-10, max_iter=1000, line_search='wolfe')
    m.solve()

    errors = []
    for xk, Hk in zip(m.x_history, m.H_history):
        Ginv = np.linalg.inv(prob.hess(xk))
        errors.append(np.linalg.norm(Hk-Ginv, 'fro') / np.linalg.norm(Ginv, 'fro'))
    plt.semilogy(errors, 'o-', label=label)

    Gk = prob.hess(m.x_history[-1])
    print(label, "last eig value:", np.sort(np.linalg.eigvals(m.H_history[-1] @ Gk).real))

def task12():
    plt.figure()
    bfgs_hessian_quality(optimization(rosenbrock, rosen_grad, rosen_hess),
                         np.array([-1.2,1.0]),"Rosenbrock")
    bfgs_hessian_quality(optimization(chebyquad, gradchebyquad),
                         np.linspace(0,1,8), "chebyquad n=8")
    plt.xlabel("iteration k")
    plt.ylabel(r"$\|H^{(k)} - G(x^{(k)})^{-1}\|_F \, / \, \|G(x^{(k)})^{-1}\|_F$")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    task9()

