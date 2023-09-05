import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy import io
import numpy as np
from tqdm import tqdm
import sys
import os

sys.path.insert(0, "./designs")

from design_parser import parse_design

# create a colormap from black to white to light blue
traa_blue = [91 / 255, 206 / 255, 250 / 255]
cdict = {
    "red": [
        [0.0] + [0.0] * 2,
        [0.5] + [1.0] * 2,
        [1.0] + [traa_blue[0]] * 2,
    ],
    "green": [
        [0.0] + [0.0] * 2,
        [0.5] + [1.0] * 2,
        [1.0] + [traa_blue[1]] * 2,
    ],
    "blue": [
        [0.0] + [0.0] * 2,
        [0.5] + [1.0] * 2,
        [1.0] + [traa_blue[2]] * 2,
    ],
}
black2blue = colors.LinearSegmentedColormap("testCmap", segmentdata=cdict)


def plot_design(design, data_path, N, eta):
    data = io.loadmat(data_path)["data"]

    parameters, *_ = parse_design(os.path.join("designs", design) + ".json")
    w, h = parameters.width, parameters.height

    N = int(np.sqrt(data.size / (w * h)))
    Nx, Ny = int(N * w), int(N * h)
    data = data.reshape((Ny, Nx))

    # rescale so design is between 0 and 1 (why is it not?)
    minimum, maximum = np.min(data), np.max(data)
    data = (data - minimum) / (maximum - minimum)

    plt.figure(figsize=(6.4 * w/h, 4.8))

    X, Y = np.meshgrid(np.linspace(0, w, Nx), np.linspace(0, h, Ny))
    plt.pcolormesh(X, Y, data, cmap=black2blue)
    plt.colorbar(label=r"$\rho(x, y)$ []")
    plt.xlim(0, w)
    plt.ylim(0, h)
    plt.gca().set_aspect("equal", "box")

    plt.xlabel("$x$ []")
    plt.ylabel("$y$ []")

    output_file = os.path.join("output", design, "figures", f"{N=}_{eta=}") + ".png"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    designs = []
    for design in os.listdir("output"):
        data_folder = os.path.join("output", design, "data")
        if not os.path.isdir(data_folder):
            continue

        for data in os.listdir(data_folder):
            data_path = os.path.join(data_folder, data)
            if not os.path.isfile(data_path):
                continue

            N_str, eta_str = data[:-4].split("_")
            N = int(N_str.split("=")[1])
            eta = float(eta_str.split("=")[1])

            designs.append((design, data_path, N, eta))

    for design in tqdm(designs):
        plot_design(*design)
