import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy import io
import numpy as np

# create a colormap from black to light blue
traa_blue = [91 / 255, 206 / 255, 250 / 255]
cdict = {
    "red": [[0.0] + [0.0] * 2, [1.0] + [traa_blue[0]] * 2],
    "green": [[0.0] + [0.0] * 2, [1.0] + [traa_blue[1]] * 2],
    "blue": [[0.0] + [0.0] * 2, [1.0] + [traa_blue[2]] * 2],
}
black2blue = colors.LinearSegmentedColormap("testCmap", segmentdata=cdict)


def plot_design(N, eta):
    data = io.loadmat(f"output/data/design_{N=}_{eta=}.mat")
    design = data["data"]
    w, h = data["w"][0, 0], data["h"][0, 0]

    Nx, Ny = int(N * w), int(N * h)
    design = design.reshape((Ny, Nx))

    # rescale so design is between 0 and 1 (why is it not?)
    minimum, maximum = np.min(design), np.max(design)
    design = (design - minimum) / (maximum - minimum)

    X, Y = np.meshgrid(np.linspace(0, w, Nx), np.linspace(0, h, Ny))
    plt.pcolormesh(X, Y, design, cmap=black2blue)
    plt.colorbar(label=r"$\rho(x, y)$ []")

    plt.xlabel("$x$ []")
    plt.ylabel("$y$ []")
    plt.savefig(f"output/figures/design_{N=}_{eta=}.png", dpi=200)
    plt.clf()


plot_design(40, 0)
plot_design(40, 40)
plot_design(40, 200)
plot_design(40, 1000)
