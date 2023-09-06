from dataclasses import dataclass
import json


@dataclass
class SolverParameters:
    width: float
    height: float
    fraction: float
    objective: str


@dataclass
class Flow:
    side: str
    center: float
    length: float
    rate: float

    def __iter__(self):
        return iter((self.side, self.center, self.length, self.rate))


@dataclass
class Sides:
    sides: list[str]


@dataclass
class Region:
    center: tuple[float, float]
    size: tuple[float, float]


def parse_design(filename: str):
    with open(filename, "r") as design_file:
        design = json.load(design_file)

    flows: list[Flow] = []
    legal_directions = ["left", "right", "top", "bottom"]
    for flow_dict in design["flows"]:
        if not flow_dict["side"] in legal_directions:
            print(
                f"Error: Got design with malformed flow direction: '{flow_dict['side']}'"
            )
            print(f"Legal directions are: {', '.join(legal_directions)}")
            exit(1)
        flows.append(
            Flow(
                flow_dict["side"],
                flow_dict["center"],
                flow_dict["length"],
                flow_dict["rate"],
            )
        )

    if len(design.get("zero_pressure", [])) == 0:
        total_flow = 0
        for flow in flows:
            total_flow += flow.rate * flow.length

        if abs(total_flow) > 1e-14:
            print(f"Error: Illegal design: total flow is {total_flow}, not 0!")
            exit(1)

    legal_objectives = ["minimize_power", "maximize_flow"]
    if not design["objective"] in legal_objectives:
        print(f"Error: Got design with malformed objective: '{design['objective']}'")
        print(f"Legal objectives are: {', '.join(legal_objectives)}")
        exit(1)

    parameters = SolverParameters(
        design["width"], design["height"], design["fraction"], design["objective"]
    )

    no_slip = None
    if design.get("no_slip"):
        sides = []
        for side in design["no_slip"]:
            if not side in legal_directions:
                print(f"Error: Got design with malformed no slip side: '{side}'")
                print(f"Legal sides are: {', '.join(legal_directions)}")
                exit(1)
            sides.append(side)
        no_slip = Sides(sides)

    zero_pressure = None
    if design.get("zero_pressure"):
        sides = []
        for side in design["zero_pressure"]:
            if not side in legal_directions:
                print(f"Error: Got design with malformed zero pressure side: '{side}'")
                print(f"Legal sides are: {', '.join(legal_directions)}")
                exit(1)
            sides.append(side)
        zero_pressure = Sides(sides)

    max_region = None
    if design.get("max_region"):
        center = design["max_region"]["center"]
        cx, cy = float(center[0]), float(center[1])

        size = design["max_region"]["size"]
        w, h = float(size[0]), float(size[1])

        max_region = Region((cx, cy), (w, h))
    elif parameters.objective == "maximize_flow":
        print("Error: Got maximize flow objective with no max region!")
        exit(1)

    return parameters, flows, no_slip, zero_pressure, max_region


if __name__ == "__main__":
    parameters, flows, no_slip, zero_pressure, max_region = parse_design(
        "designs/fluid_mechanism.json"
    )
    print(parameters)
    print(flows)
    print(no_slip)
    print(zero_pressure)
    print(max_region)
