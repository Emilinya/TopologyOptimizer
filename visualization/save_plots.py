import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy import io
import numpy as np

# create a colormap from black to light blue
traa_blue = [91 / 255, 206 / 255, 250 / 255]
cdict = {
    "red": [[0.0] + [0.0] * 2, [1.0] + [traa_blue[0]] * 2],
    "green": [[0.0], [0.0] * 2, [1.0], [traa_blue[1]] * 2],
    "blue": [[0.0], [0.0] * 2, [1.0], [traa_blue[2]] * 2],
}
black2blue = colors.LinearSegmentedColormap("testCmap", segmentdata=cdict)


def plot_domain(N, i):
    identifier = str(N) + "_" + str(i + 1)
    data = io.loadmat("output/data/matlab_controls_" + identifier + ".mat")["data"]

    # reshape with hardcoded delta, yay ...
    delta = 1.5
    data = data.reshape((N, int(N * delta)))

    # rescale so data is between 0 and 1 (why is it not?)
    minimum, maximum = np.min(data), np.max(data)
    data = (data - minimum) / (maximum - minimum)

    X, Y = np.meshgrid(np.linspace(0, delta, int(N * delta)), np.linspace(0, 1, N))
    plt.pcolormesh(X, Y, data, cmap=black2blue)
    plt.colorbar(label=r"$\rho(x, y)$ []")

    plt.xlabel("$x$ []")
    plt.ylabel("$y$ []")
    plt.savefig("output/figures/" + identifier + ".png", dpi=200)
    plt.clf()


plot_domain(40, 0)
plot_domain(40, 1)
plot_domain(40, 2)
plot_domain(40, 3)
