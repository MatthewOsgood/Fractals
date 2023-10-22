import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def julia(mesh, num_iter=10, radius=2, c=-1, f=np.square):
    z = mesh.copy()
    diverge_len = np.zeros(z.shape)  # tracks number of iterations applied
    for i in range(num_iter):
        conv_mask = np.abs(z) < radius  # check which elements have diverged
        z[conv_mask] = f(z[conv_mask]) + c  # iterate the ones that haven't diverged
        diverge_len[conv_mask] += 1  # tally the ones that haven't diverged yet
    return diverge_len


def mandelbrot(mesh, num_iter=10, radius=2):
    c = mesh.copy()
    z = np.zeros(c.shape, dtype=np.complex128)
    diverge_len = np.zeros(z.shape)
    for i in range(num_iter):
        conv_mask = np.abs(z) < radius
        z[conv_mask] = np.square(z[conv_mask]) + c[conv_mask]
        diverge_len[conv_mask] += 1
    return diverge_len


def graph(output):
    figure = plt.figure(figsize=(6, 6))
    ax = plt.axes()
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    im = ax.imshow(output, extent=(-2, 2, -2, 2))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=.1)
    plt.colorbar(im, cax=cax, label="Number of Iterations")

    plt.show()


def main():
    x, y = np.meshgrid(np.linspace(-2, 2, 1000), np.linspace(-2, 2, 1000))
    complex_plane = x + (1j * y)
    output1 = julia(complex_plane, num_iter=15, c=-1, f=lambda z : np.sin(np.tan(z)))
    output2 = mandelbrot(complex_plane, num_iter=100)
    graph(output1)
    graph(output2)


if __name__ == "__main__":
    main()
