import os
import sys

sys.path.insert(0, "./src")

from fluid_solver import FluidSolver


if __name__ == "__main__":
    if len(sys.argv) == 3:
        got_error = False

        design_file = sys.argv[1]
        if not os.path.isfile(design_file):
            print("Got a design path that is not a file!")
            got_error = True

        N = sys.argv[2]
        try:
            N = int(N)
        except:
            print("Got a domain size that is not an integer!")
            got_error = True

        if got_error:
            print("This program is used as follows:")
            print(f"  python3 {sys.argv[0]} <design file> <domain size (N)>")
        else:
            solver = FluidSolver(design_file, N)
            solver.solve()
    else:
        print("Got an invalid number of arguments. This program is used as follows:")
        print(f"  python3 {sys.argv[0]} <design file> <domain size (N)>")
