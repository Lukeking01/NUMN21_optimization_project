import numpy as np
from scipy.optimize import line_search, minimize_scalar



def exact_line_search(problem, x, s, alpha_max=100.0):
    phi = lambda a: problem.f(x + a*s)
    res = minimize_scalar(phi, bounds=(0.0, alpha_max), method='bounded',
                        options={'xatol': 1e-12})
    return res.x

#task 1
class optimization:
    def __init__(self, f, grad=None, hess=None):
        self.f=f
        self._grad=grad
        self._hess=hess

    def grad(self,x):
        """ central difference """
        if self._grad is not None:
            return np.asarray(self._grad(x), dtype=float)
        x=np.asarray(x,dtype=float)
        n=x.size
        g=np.zeros(n)
        for i in range(n):
            ei = np.zeros(n) #e_i:th dimension
            ei[i] = 1.0
            h = 1e-5
            g[i] = (self.f(x+ h*ei) - self.f(x- h*ei)) / (2*h)
        return g

    def hess(self,x):
        if self._hess is not None:
            return np.asarray(self._hess(x), dtype=float)
        x=np.asarray(x,dtype=float)
        n=x.size
        H=np.zeros((n,n))
        for i in range(n):
            ei= np.zeros(n)
            ei[i]=1.0
            h= 1e-5
            H[:,i] = (self.grad(x+ h*ei) - self.grad(x- h*ei))/ (2*h)
        return 0.5 * (H+H.T) #symmetrizing step

#task 2
class optimizationMethod:
    def __init__(self, problem, x0, tol=1e-8, max_iter=1000, H0=None, line_search='wolfe'):
        self.problem=problem
        self.x0=np.asarray(x0, dtype=float).copy()
        self.tol=tol
        self.max_iter=max_iter

        self.H0=H0
        self.line_search=line_search

        #history for later tasks
        self.x_history=[]
        self.f_history=[]
        self.g_history=[]
        self.H_history=[]
        self.alpha_history=[]


        #override by specific methods
    def update_H(self, H, delta, gamma):
        raise NotImplementedError("subclass must implement update_H")

        #line-search dispatcher
    def compute_step_size(self,x,s):
        if self.line_search == 'wolfe':
            return line_search_wolfe(self.problem,x,s)
        if self.line_search == 'exact':
            return exact_line_search(self.problem,x,s)
        return 1.0

        #same loop for all methods
    def solve(self):
        x = self.x0.copy()
        n= x.size
        H = np.eye(n) if self.H0 is None else self.H0.copy()

        self.x_history=[x.copy()]
        self.H_history=[H.copy()]
        self.f_history=[self.problem.f(x)]
        self.g_history=[self.problem.grad(x)]
        self.alpha_history=[]

        for i in range(self.max_iter):
            g= self.g_history[-1]

            if np.linalg.norm(g) < self.tol: #residual stopping criteria
                break

                # Quasi-Newton  
            s=-H @ g

            if g@s >= 0:
                s = -g
                H = np.eye(n)

                 #step size
            alpha = self.compute_step_size(x,s)
            if alpha == 0.0:
                break
            self.alpha_history.append(alpha)

            x_new= x+ alpha*s
            g_new = self.problem.grad(x_new)

            delta= x_new-x
            gamma= g_new -g

            # update H
            H = self.update_H(H,delta, gamma)

            x=x_new
            self.x_history.append(x.copy())
            self.H_history.append(H.copy())
            self.f_history.append(self.problem.f(x))
            self.g_history.append(g_new)
                #Cauchy
            if np.linalg.norm(alpha*s) < self.tol:
                break
        return x


#task9
class GoodBroyden(optimizationMethod):
    def update_H(self,H,delta,gamma):
        Hg = H@gamma
        denom = delta@Hg
        if abs(denom) < 1e-12:
            return H
        return H + np.outer(delta-Hg, delta@H)/denom

class BadBroyden(optimizationMethod):
    def update_H(self,H,delta,gamma):
        denom = gamma @ gamma
        if denom < 1e-12:
            return H
        return H + np.outer(delta-H@gamma, gamma)/denom

class symmetricBroyden(optimizationMethod):
    def update_H(self,H,delta,gamma):
        u = delta-H@gamma
        denom = u @ gamma
        if abs(denom) < 1e-8 * np.linalg.norm(u) * np.linalg.norm(gamma):
            return H
        return H+ np.outer(u,u)/denom
    
class DFP(optimizationMethod):
    def update_H(self, H, delta, gamma):
        dg = delta @ gamma
        Hg = H @ gamma
        gHg = gamma @ Hg
        if dg < 1e-12 or gHg < 1e-12:
            return H
        return H + np.outer(delta, delta) / dg - np.outer(Hg, Hg) / gHg

class BFGS(optimizationMethod):
    def update_H(self, H, delta, gamma):
        dg = delta @ gamma
        if dg < 1e-12:
            return H
        Hg = H @ gamma
        return (H
                + (1.0 + gamma @ Hg / dg) * np.outer(delta, delta) / dg
                - (np.outer(delta, Hg) + np.outer(Hg, delta)) / dg)




#Task 3 and 4
class NewtonMethod:
    """ classical newton method """
    def __init__(self, problem, x0, tol=1e-8, max_iter=100, exact_line_search=True):
        self.problem = problem
        self.x0 = x0
        self.tol = tol
        self.max_iter = max_iter
        self.exact_line_search = exact_line_search

        self.x_history=[]
        self.f_history=[]
        self.g_history=[]
        self.alpha_history=[]

    def _exact_line_search(self, x, s, alpha_max=100.0):
        phi = lambda a: self.problem.f(x+a*s)
        res = minimize_scalar(phi, bounds=(0.0, alpha_max), method='bounded', options={'xatol':1e-12})
        return res.x

    def solve(self):
        x= self.x0.copy()
        self.x_history = [x.copy()]
        self.f_history = [self.problem.f(x)]
        self.g_history = [self.problem.grad(x)]
        self.alpha_history = []

        for i in range(self.max_iter):

            print(f"iter {i:2d}: x = {x}, f = {self.problem.f(x):.4e}")
            

            g = self.problem.grad(x)



            gnorm = np.linalg.norm(g)

            if gnorm < self.tol:
                print("  -> break: residual criterion")
                break

            G = self.problem.hess(x)

            try:
                s = np.linalg.solve(G, -g)
            except np.linalg.LinAlgError: # if singular hessian
                s = -g


            if self.exact_line_search:
                alpha = self._exact_line_search(x,s)
            else:
                alpha = line_search_wolfe(self.problem, x,s)
            self.alpha_history.append(alpha)
            
            x_new= x + s*alpha

            self.x_history.append(x_new.copy())
            self.f_history.append(self.problem.f(x_new))
            self.g_history.append(self.problem.grad(x_new))

            step_norm = np.linalg.norm(x_new-x)
            x=x_new

            #Cauchy stopping criterion
            if step_norm < self.tol:
                break
        return x


#task6
def line_search_wolfe(problem,x,s,alpha1=1.0, rho=0.01, sigma=0.1,tau1=9.0, alpha_max=1e3, max_iter=50):

    x = np.asanyarray(x, dtype=float)
    s = np.asanyarray(s, dtype=float)

    phi = lambda a: problem.f(x+a*s)
    dphi = lambda a: problem.grad(x+a*s)@s

    phi0, dphi0 = phi(0.0), dphi(0.0)
    if dphi0 >= 0:
        raise ValueError('s is not in descent direction')

    def armijo_fails(a,p):
        return p> phi0+rho*a*dphi0

    def curvature_ok(d):
        return abs(d) <= -sigma *dphi0

    #sectioning
    def sectioning(a,b,phi_a):
        alpha = 0.5*(a+b)
        for i in range(max_iter):
            alpha=0.5*(a+b)
            p = phi(alpha)
            if armijo_fails(alpha,p) or p>= phi_a:
                b=alpha
            else:
                d = dphi(alpha)
                if curvature_ok(d):
                    return alpha
                if (b-a) * d>=0:
                    b=a
                a,phi_a = alpha, p
            if abs(b-a) < 1e-14:
                break
        return alpha
    #bracketing

    a_prev, phi_prev = 0.0, phi0
    alpha = min(alpha1, alpha_max)
    for i in range(max_iter):
        p = phi(alpha)
        if armijo_fails(alpha,p) or p >= phi_prev:
            return sectioning(a_prev, alpha, phi_prev)
        d = dphi(alpha)
        if curvature_ok(d):
            return alpha
        if d>=0:
            return sectioning(alpha,a_prev,p)
        if alpha >= alpha_max:
            return alpha
        new = min(alpha_max, alpha+tau1*(alpha-a_prev))
        a_prev, phi_prev = alpha, p
        alpha = new

    return alpha

