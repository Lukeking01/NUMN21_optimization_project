import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def newton_fractal(width=1200, height=1200, max_iter=40,
                   re_min=-1.5, re_max=1.5, im_min=-1.5, im_max=1.5):
    
    re = np.linspace(re_min, re_max, width)
    im = np.linspace(im_min, im_max, height)
    z = re[np.newaxis, :] + 1j * im[:, np.newaxis]
    
    # roots of z³ - 1 = 0
    roots = np.array([1.0,
                      -0.5 + 0.8660254037844386j,
                      -0.5 - 0.8660254037844386j])
    
    # colour by which root we converge to + how fast
    epsilon = 1e-6
    colors = np.zeros(z.shape)
    iterations = np.zeros(z.shape)
    
    for i in range(max_iter):
        dz = (z**3 - 1) / (3 * z**2)
        z = z - dz
        
        # record when we get close to a root
        for k, root in enumerate(roots):
            mask = (np.abs(z - root) < epsilon) & (colors == 0)
            colors[mask] = k + 1
            iterations[mask] = i
    
    # points that never converged
    colors[colors == 0] = 4
    
    return colors, iterations

# Generate high-resolution fractal
print("Generating fractal...")
colors, iters = newton_fractal(width=400, height=400, max_iter=50)

# Beautiful colour map
cmap = LinearSegmentedColormap.from_list("newton", 
    ["#000000", "#ff0055", "#00ffaa", "#5500ff", "#222222"])

plt.figure(figsize=(12, 12), dpi=150)
plt.imshow(colors, extent=[-1.5, 1.5, -1.5, 1.5],
           cmap=cmap, origin='lower', interpolation='nearest')
plt.axis('off')
plt.title("Newton fractal for $z^3 - 1 = 0$", fontsize=16, pad=10)
plt.tight_layout()
# plt.savefig("newton_fractal.png", dpi=200, bbox_inches='tight', pad_inches=0.1)
plt.show()