from matplotlib.ticker import MaxNLocator
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2 as cv
from scipy.optimize import curve_fit


def tanh_func(x, a, b, c, d):
    return a * np.tanh(b * (x + c)) + d


def get_cfd_values(path):
    mesh = pv.read(path)

    x = mesh.points[:, 0]
    y = mesh.points[:, 1]
    density = mesh.point_data["Density"]

    mesh_g = mesh.compute_derivative(
    scalars="Density",
    gradient="grad_density",
    preference="point",
)
    grad = mesh_g["grad_density"]
    grad_mag = np.linalg.norm(grad, axis=1)

    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    density = density[idx]
    grad_mag = grad_mag[idx]

    # Line of best fit
    initial_guess = [1.0, 1.0, np.mean(x), np.mean(density)]
    params, _ = curve_fit(tanh_func, x, density, p0=initial_guess)
    a, b, c, d = params

    x_fit = np.linspace(min(x), max(x), len(x))
    density_fit = tanh_func(x_fit, a, b, c, d)

    # density_prime = np.diff(density, prepend=x[0]) / np.diff(x, prepend=x[0])

    return x, y, density, grad_mag


def get_straight_values(x, y, density_prime):
    new_x = []
    line = []
    for i in range(len(x)):
        if y[i] == 0:
            line.append(density_prime[i])
            new_x.append(x[i])

    return new_x, line


u = np.load("u_HS.npy")
v = np.load("v_HS.npy") # V is upwards, U is rightwards
mask = cv.imread("Correlable_pics/BOS_12_11_ref_masked.tif", cv.IMREAD_GRAYSCALE)

x, y, density, density_prime = get_cfd_values("Wavelet_noise_experiments/220/BOS_12_11_1/flow_MUSCL.vtu")

new_x, line = get_straight_values(x, y, density_prime)

plt.scatter(new_x, line, s=10)
plt.xlabel(r"$x$ [$m$]")
plt.ylabel(r"Density Gradient [$kg/m^2$]")
plt.title(r"Density Gradient ($y=0$)")
plt.savefig("Density_Gradient_y0.png")
plt.show()
plt.close()

# Density
fig, ax = plt.subplots(figsize=(16, 4))
sc = ax.scatter(x, y, c=density, s=10)

ax.set_xlabel(r"$x$ [$m$]")
ax.set_ylabel(r"$y$ [$m$]")
ax.set_title("Density")
ax.set_aspect('equal', adjustable='box')

cbar = fig.colorbar(sc, ax=ax, orientation='horizontal', pad=0.25, fraction=0.1, aspect=60)
cbar.set_label(r"[$kg/m^3$]")

fig.savefig("Density.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)

# Density Gradient
fig, ax = plt.subplots(figsize=(16, 4))
sc = ax.scatter(x, y, c=density_prime, s=10)

ax.set_xlabel(r"$x$ [$m$]")
ax.set_ylabel(r"$y$ [$m$]")
ax.set_title("Density Gradient")
ax.set_aspect('equal', adjustable='box')

cbar = fig.colorbar(sc, ax=ax, orientation='horizontal', pad=0.25, fraction=0.1, aspect=60)
cbar.set_label(r"[$kg/m^2$]")

fig.savefig("Density_Gradient.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)
