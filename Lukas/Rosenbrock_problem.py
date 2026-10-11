
import numpy as np

def rosenbrock(x):

    x1 = x[0]
    x2 = x[1]

    return (100.0 * (x2 - x1**2)**2 + (1.0 - x1)**2)


def rosenbrock_gradient(x):

    x1 = x[0]
    x2 = x[1]

    return np.array([-400.0 * x1 * (x2 - x1**2)- 2.0 * (1.0 - x1)
        ,
        200.0 * (x2 - x1**2)])
    

def plot_rosenbrock_path(method, ax, xlim=(-1.5, 2.0), ylim=(-0.5, 3.5)):
 
    x_values = np.linspace(xlim[0],xlim[1],150)

    y_values = np.linspace(ylim[0],ylim[1],150)

    X, Y = np.meshgrid(x_values,y_values)

    Z = rosenbrock(np.array([X,Y]))

    # plt.figure(figsize=(8, 6))

    levels = np.logspace(-1,3,14)

    ax.contour(X,Y,Z,levels=levels)

    history = np.array(method.history)

    ax.plot(history[:, 0],history[:, 1],"o-",label="optimization path", color="green")
    ax.plot(history[0,0], history[0,1],"^",markersize=12,label="initial guess", color="blue")
    ax.plot(1.0,1.0,"*",markersize=12,label="minimum (1, 1)", color="yellow")

    # ax.set_xlabel("x1")
    # ax.set_ylabel("x2")
    ax.set_title(method.__class__.__name__ + " with " + method.line_search.__class__.__name__)
    ax.legend()
    # ax.grid(True)