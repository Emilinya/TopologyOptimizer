import os
import sys

sys.path.insert(0, "./topopt")

from topopt import FluidSolver


def print_use():
    print(
        f"  python3 {sys.argv[0]} default                          "
        + "Run program with default (hardcoded) values"
    )
    print(
        f"  python3 {sys.argv[0]} <design file> <domain size (N)>  "
        + "Run program with custom parameters defined in a design file"
    )


if __name__ == "__main__":
    if len(sys.argv) == 2:
        N = 10
        solver = FluidSolver("designs/pipe_bend.json", N)
        solver.solve()
    elif len(sys.argv) == 3:
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
            print_use()
        else:
            solver = FluidSolver(design_file, N)
            solver.solve()
    else:
        print("Got an invalid number of arguments. This program is used as follows:")
        print_use()
