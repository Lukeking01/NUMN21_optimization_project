import numpy as np
import matplotlib.pyplot as plt

def draw_contour(
        fn, steps=[],
        print_labels = False,
        n_levels = 7, ls_off = 100, le_off = 10,
    ):
    """
    Draw a contour plot of the function. Expects two inputs to one output.
    """
    
    # Compute dimensions of the contour graph from the steps, with a suitable fallback
    x_range = [-2, 2]
    y_range = [-1.5, 4]

    if len(steps) > 0:
        padding = 4
        nd_steps = np.array(steps)
        x, y = nd_steps[:,0], nd_steps[:,1]
        x_range = [np.min(x) - padding, np.max(x) + padding]
        y_range = [np.min(y) - padding, np.max(y) + padding]
    
    # Compute 2D grid over domain
    x_vals = np.linspace(*x_range, 100)
    y_vals = np.linspace(*y_range, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = fn([X, Y])

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))

    # Contour levels (log-space)
    levels = np.logspace(np.log10(np.min(Z) * ls_off), np.log10(np.max(Z) * le_off), n_levels)
    
    # Draw and optionally label contour levels
    contour_lines = ax.contour(X, Y, Z, levels=levels, colors="black", alpha=1, linewidths=0.8)
    if print_labels:
        ax.clabel(contour_lines, inline=True, fontsize=6, fmt="%.3f")

    # Add scatter of steps taken
    if len(steps) > 0:
        nd_steps = np.array(steps)
        x, y = nd_steps[:,0], nd_steps[:,1]
        ax.plot(x, y, color='blue', zorder=4)
        ax.scatter(x, y, color='blue', s=42, zorder=5, label="Steps")
        plt.scatter(
            x[-1], y[-1],
            color='blue',          # Interior fill color
            edgecolor='red',       # Outline color
            linewidth=2.5,         # Outline thickness
            s=50,                  # Larger marker size
            zorder=10,             # Ensures it stays on top
            label='Final Step'
        )

    # Labels and styling
    ax.set_title("2D Contour Graph", fontsize=16, weight=700, pad=12)
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

def draw_error_grid(errors):
    """
    (Stolen func, not from me :p)
    Draw the errors from approximating the Hessian to a grid plot.
    """
    
    plt.figure(figsize=(8, 5))
    plt.semilogy(errors, "bo-", linewidth=2, markersize=6)
    plt.title("Hessian (inv) Approximation Quality")
    plt.xlabel("Iteration $k$")
    plt.ylabel("Frobenius Norm Error")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.show()

def rosenbrock_fn(x):
    return 100 * ((x[1] - x[0]**2)**2) + (1 - x[0])**2

if __name__ == "__main__":
    # Few labeled levels
    draw_contour(rosenbrock_fn, print_labels=True, n_levels=7, ls_off=100, le_off=10)
    # Many unlabeled levels
    draw_contour(rosenbrock_fn, print_labels=False, n_levels=24, ls_off=100, le_off=10)
