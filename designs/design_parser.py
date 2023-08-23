import matplotlib.pyplot as plt
import numpy as np
import json


def create_discretizer(partition_size):
    def discretize(size):
        return int(size / partition_size)
    return discretize


def flow_profile(t, rate, length):
    return rate*(1 - (2*t/length)**2)


with open("designs/diffuser.json", "r") as design_file:
    design = json.load(design_file)

total_flow = 0
for flow in design["flows"]:
    total_flow += flow["rate"] * flow["length"]

if abs(total_flow) > 1e-14:
    print(f"Illegal design: total flow is {total_flow}, not 0!")
    exit(1)

partition_size = 1e-2
discretize = create_discretizer(partition_size)

width, height = design["width"], design["height"]
x_partitions, y_partitions = discretize(width), discretize(height)

boundry_flow_matrix = np.zeros((y_partitions, x_partitions))
for flow in design["flows"]:
    rate, length, center = flow["rate"], flow["length"], flow["center"]

    left = discretize(center) - int(discretize(length) / 2)
    right = left + discretize(length)
    t_ray = np.linspace(-length/2, length/2, discretize(length))
    flow_ray = flow_profile(t_ray, rate, length)

    if flow["side"] == "top":
        boundry_flow_matrix[0, left:right] = flow_ray
    elif flow["side"] == "left":
        boundry_flow_matrix[left:right, 0] = flow_ray
    elif flow["side"] == "bottom":
        boundry_flow_matrix[-1, left:right] = flow_ray
    elif flow["side"] == "right":
        boundry_flow_matrix[left:right, -1] = flow_ray
    else:
        print(f"Illegal design: unknown flow direction {flow['size']}")
        exit(1)

plt.imshow(boundry_flow_matrix)
plt.colorbar()
plt.show()
