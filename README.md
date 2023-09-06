[![GNU GPLv3 License](https://img.shields.io/github/license/Emilinya/TopologyOptimizer)](https://choosealicense.com/licenses/gpl-3.0/)
[![Test TopologyOptimizer](https://github.com/Emilinya/TopologyOptimizer/actions/workflows/test-TopOpt.yml/badge.svg?style=plastic)](https://github.com/Emilinya/TopologyOptimizer/actions/workflows/test-TopOpt.yml)

# TopologyOptimizer 

This program is based on [TopOpt](https://github.com/JohannesHaubner/TopOpt) by Johannes Haubner, Franziska Neumann and Michael Ulbrich. 

## Usage/Examples

### Running
The program is run using `run.py`. This program takes in two command line arguments; a design file and a domain size. The folder `designs` contains some design files, and you can easily make a custom design using those files as a template. If the design is `path/to/design_example.json`, the output of the program is saved to `output/design_example/data`. The produced data can be visualized with `plot.py`, which automatically reads all the data files, and produces corresponding figures in `output/design_example/figures`

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

## Running Tests
In theory, tests exists, but I have not gotten them to run. The original repository said "To run tests, run the following command:"

```bash
pytest
```

But doing this with the docker image simply says `bash: pytest: command not found`.

## License

[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)
