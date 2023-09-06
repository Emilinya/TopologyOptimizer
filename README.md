[![GNU GPLv3 License](https://img.shields.io/github/license/Emilinya/TopologyOptimizer)](https://choosealicense.com/licenses/gpl-3.0/)
[![Test TopologyOptimizer](https://github.com/Emilinya/TopologyOptimizer/actions/workflows/test-TopOpt.yml/badge.svg?style=plastic)](https://github.com/Emilinya/TopologyOptimizer/actions/workflows/test-TopOpt.yml)

# TopologyOptimizer 

This program is based on [TopOpt](https://github.com/JohannesHaubner/TopOpt) by Johannes Haubner, Franziska Neumann and Michael Ulbrich. 

## Usage/Examples

### Running
The program is run using `run.py`. This program takes in two command line arguments; a design file and a domain size. The folder `designs` contains some design files, and you can easily make a custom design using those files as a template. If the design is `path/to/design.json`, the output of the program is saved to `output/design/data`. The produced data can be visualized with `plot.py`, which automatically reads all the data files, and produces corresponding figures in `output/design/figures`. `plot.py` can also take a list of designs as an argument to limit which designs it will plot. For istance,
```bash
python3 plot.py design1 design2
```
will only create figures from `output/design1/data` and `output/design2/data`.

### Docker
TopologyOptimizer uses fenics-dolfin, which is not available on Windows. The TopOpt project includes a docker image which makes running the program on Windows easy. As it is not my docker image, I can't guarantee that it will work forever. To use the Docker, simply run

```bash
docker pull ghcr.io/johanneshaubner/topopt:latest

cd topopt
docker run -it -v $(pwd):/topopt ghcr.io/johanneshaubner/topopt
```

Then, in the Docker container, you can run the program as normal:
```bash
python3 run.py designs/twin_pipe 40
```

### Conda
If you are using Linux (or wsl), you can run the program using conda. I have not tested that this works, so the following text is simply copied from the TopOpt repository:

```bash
conda env create -f environment.yml --experimental-solver=libmamba
conda activate topopt

cd topopt
python3 topopt.py

conda deactivate topopt
```

It has to be ensured that the [conda-libmamba-solver](https://github.com/conda-incubator/conda-libmamba-solver) is installed.

For practical problems it is furthermore necessary to link IPOPT against HSL when compiling (see comment in http://www.dolfin-adjoint.org/en/release/documentation/stokes-topology/stokes-topology.html).

For running the MMA examples, it is required to clone the GitHub repository https://github.com/arjendeetman/GCMMA-MMA-Python into the folder mma/MMA_Python.

## Design file format
The design files are written in json, and the settings are:
1. objective \
    Allowed values: "minimize_power" or "maximize_flow" \
    Description: Descides the objective function to be minimized.
2. width \
    Allowed values: `float` \
    Description: The width of the domain
3. height \
    Allowed values: `float` \
    Description: The height of the domain
4. fraction \
    Allowed values: `float` \
    Description: The fraction of the domain that is allowed to be empty, the volume fraction.
5. flows \
    Allowed values: list of flows, where a flow has the following values:
    - side: "left", "right", "top", or "bottom"
    - center: `float`
    - length: `float`
    - rate: `float`

    Description: Defines the boundry conditions on the velocity field. Each flow describes a parabolic flow pattern. A positive rate indicates flow into the domain, and a negative flow pattern indicates flow out of the domain.
7. max_region \
    Allowed values: 
    - center: `(float, float)`
    - size: `(0, 0.05)`

    Description: The region where you want to maximize flow. Mandatory for the "maximize_flow" objective, does nothing for the "minimize_power" objective. The desired flow direction is currently hardcoded to (-1, 0).
6. no_slip (optional) \
    Allowed values: "left", "right", "top", or "bottom" \
    Description: The sides where there is no flow (velocity is 0). Defaults to all sides with no defined flow.
7. zero_pressure (optional) \
    Allowed values: "left", "right", "top", or "bottom" \
    Description: The sides where there is no pressure. If not set, pressure is 0 at (0, 0).

## Running Tests
In theory, tests exists, but I have not gotten them to run. The original repository said "To run tests, run the following command:"

```bash
pytest
```

But doing this with the docker image simply says `bash: pytest: command not found`.

## License

[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)
