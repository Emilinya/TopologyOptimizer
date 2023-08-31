from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import json


@dataclass
class SolverParameters:
    width: float
    height: float
    fraction: float


@dataclass
class Flow:
    side: str
    center: float
    length: float
    rate: float

    def __iter__(self):
        return iter((self.side, self.center, self.length, self.rate))


def parse_design(filename):
    with open(filename, "r") as design_file:
        design = json.load(design_file)

    flows = []
    for flow_dict in design["flows"]:
        flows.append(
            Flow(
                flow_dict["side"],
                flow_dict["center"],
                flow_dict["length"],
                flow_dict["rate"],
            )
        )

    total_flow = 0
    for flow in flows:
        total_flow += flow.rate * flow.length

    if abs(total_flow) > 1e-14:
        print(f"Illegal design: total flow is {total_flow}, not 0!")
        exit(1)

    parameters = SolverParameters(design["width"], design["height"], design["fraction"])

    return parameters, flows

if __name__ == "__main__":
    parameters, flows = parse_design("designs/pipe_bend.json")
    print(parameters)
    print(flows)
