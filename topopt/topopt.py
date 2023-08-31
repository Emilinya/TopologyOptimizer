import dolfin as df
import dolfin_adjoint as dfa
from scipy import io
import numpy as np
import ufl
import sys
import os

df.set_log_level(df.LogLevel.ERROR)

import Hs_regularization as Hs_reg
from ipopt_solver import IPOPTSolver, IPOPTProblem
from preprocessing import Preprocessing

sys.path.insert(0, "./designs")

from design_parser import parse_design

try:
    from pyadjoint import ipopt  # noqa: F401
except ImportError:
    print(
        """This example depends on IPOPT and Python ipopt bindings. \
        When compiling IPOPT, make sure to link against HSL, as it \
        is a necessity for practical problems."""
    )
    raise


# Define the boundary condition on velocity
class FlowBC(dfa.UserExpression):
    def __init__(self, **kwargs):
        super(FlowBC, self).__init__(kwargs)
        self.domain_size = kwargs["domain_size"]
        self.flows = kwargs["flows"]

    def get_flow(self, position, center, length, rate):
        t = position - center
        if (t > -length / 2) and (t < length / 2):
            return rate * (1 - (2 * t / length) ** 2)
        return 0

    def eval(self, values, pos):
        values[1] = 0.0
        values[0] = 0.0

        for side, center, length, rate in self.flows:
            if side == "left" and pos[0] == 0.0:
                values[0] += self.get_flow(pos[1], center, length, rate)
            elif side == "right" and pos[0] == self.domain_size[0]:
                values[0] -= self.get_flow(pos[1], center, length, rate)
            elif side == "top" and pos[1] == 0.0:
                values[1] += self.get_flow(pos[0], center, length, rate)
            elif side == "bottom" and pos[1] == self.domain_size[1]:
                values[1] -= self.get_flow(pos[0], center, length, rate)

    def value_shape(self):
        return (2,)


# Define the boundary condition on pressure
class PressureBC(df.SubDomain):
    def inside(self, x, _):
        return df.near(x[0], (0.0)) and df.near(x[1], (0.0))


class FluidSolver:
    def __init__(self, design_file, N):
        self.design_file = design_file
        parameters, flows = parse_design(self.design_file)

        # define constatns
        viscosity = dfa.Constant(1.0)
        self.alpha_max = 2.5 * viscosity / (0.01**2)

        # define domain
        self.N = N
        self.width = parameters.width
        self.height = parameters.height

        Nx, Ny = int(self.width * self.N), int(self.height * self.N)

        volume_fraction = parameters.fraction
        self.V = self.width * self.height * volume_fraction
        mesh = dfa.Mesh(
            dfa.RectangleMesh(
                df.MPI.comm_world,
                df.Point(0.0, 0.0),
                df.Point(self.width, self.height),
                Nx,
                Ny,
            )
        )

        # define solution space
        control_space = df.FunctionSpace(mesh, "DG", 0)
        velocity_space = df.VectorElement("CG", mesh.ufl_cell(), 2)
        pressure_space = df.FiniteElement("CG", mesh.ufl_cell(), 1)
        self.solution_space = df.FunctionSpace(mesh, velocity_space * pressure_space)

        # define boundary conditions
        flowBC = FlowBC(degree=2, domain_size=(self.width, self.height), flows=flows)
        pressureBC = PressureBC()
        self.boundary_conditions = [
            dfa.DirichletBC(self.solution_space.sub(0), flowBC, "on_boundary"),
            dfa.DirichletBC(
                self.solution_space.sub(1),
                dfa.Constant(0.0),
                pressureBC,
                method="pointwise",
            ),
        ]

        # create initial conditions
        k = Nx * Ny * 2
        self.x0 = (2.0 * volume_fraction - 1) * np.ones(int(k / 2))

        # preprocessing class which contains dof_to_control-mapping
        self.preprocessing = Preprocessing(self.N, control_space)
        rho = self.preprocessing.dof_to_control(self.x0)

        # get reduced objective function: rho --> j(rho)
        dfa.set_working_tape(dfa.Tape())
        w = self.forward(rho)
        (u, _) = df.split(w)

        # objective function
        J = dfa.assemble(
            0.5 * df.inner(self.alpha(rho) * u, u) * df.dx
            + 0.5 * viscosity * df.inner(df.grad(u), df.grad(u)) * df.dx
        )
        # penalty term in objective function
        J2 = dfa.assemble(
            ufl.Max(rho - 1.0, 0.0) ** 2 * df.dx + ufl.Max(-rho - 1.0, 0.0) ** 2 * df.dx
        )

        self.Js = [J, J2]
        self.control = dfa.Control(rho)
        self.Jeval = dfa.ReducedFunctional(J, self.control)
        # Note: the evaluation of the gradient can be speed up since the adjoint solve requires no pde solve
        # (see Appendix A.4)

        # constraints
        volume_constraint = 1.0 / self.V * dfa.assemble((0.5 * (rho + 1)) * df.dx) - 1.0
        spherical_constraint = dfa.assemble(
            volume_fraction / self.V * (rho * rho - 1.0) * df.dx
        )
        self.constraints = [
            dfa.ReducedFunctional(volume_constraint, self.control),
            dfa.ReducedFunctional(spherical_constraint, self.control),
        ]

    def alpha(self, rho):
        """Inverse permeability as a function of rho, equation (40)"""
        equation = self.alpha_max * (
            -1.0 / 16 * rho**4 + 3.0 / 8 * rho**2 - 0.5 * rho + 3.0 / 16
        )
        return df.conditional(
            df.gt(rho, 1.0),
            0.0,
            df.conditional(df.gt(rho, -1.0), equation, -1.0 * self.alpha_max * rho),
        )

    def forward(self, rho):
        """Solve the forward problem for a given fluid distribution rho(x)."""
        w = dfa.Function(self.solution_space)
        (u, p) = df.TrialFunctions(self.solution_space)
        (v, q) = df.TestFunctions(self.solution_space)

        F = (
            self.alpha(rho) * df.inner(u, v)
            + df.inner(df.grad(u), df.grad(v))
            + df.inner(df.grad(p), v)
            + df.inner(df.div(u), q)
        ) * df.dx

        dfa.solve(df.lhs(F) == df.rhs(F), w, bcs=self.boundary_conditions)

        return w

    def solve(self):
        # scaling of constraints for Ipopt
        scaling_constraints = [1.0, 1.0]

        # regularization parameter
        reg = 10.0

        # different weights for H_sigma matrix
        weights = [1.0, 0.01, 0.01, 0.001]
        # different penalization parameters
        etas = [0, 40, 200, 1000]
        # [[lower bound vc, upper bound vc],[lower bound sc, upper bound sc]]
        bounds = [[[0.0, 0.0], [-1.0, 0.0]]] + [[[-1e6, 0.0], [0.0, 0.0]]] * 3

        for j, (weight, eta, bound) in enumerate(zip(weights, etas, bounds)):
            # update inner product
            # consider L2-mass-matrix + weighting * Hs-matrix
            sigma = 7.0 / 16
            inner_product_matrix = Hs_reg.AssembleHs(
                self.N, self.width, sigma
            ).get_matrix(weight)

            # for performance reasons we first add J and J2 and hand the sum over to the IPOPT solver
            J_ = self.Js[0] + eta * self.Js[1]
            Jhat = [dfa.ReducedFunctional(J_, self.control)]
            # Note: the evaluation of the gradient can be speed up since the adjoint solve requires no pde solve
            # (see Appendix A.4)

            if j != 0:
                # move x0 onto sphere
                self.x0 = self.preprocessing.move_onto_sphere(
                    self.x0, self.V, self.width
                )

            # solve optimization problem
            problem = IPOPTProblem(
                Jhat,
                [1.0, eta],
                self.constraints,
                scaling_constraints,
                bound,
                self.preprocessing,
                inner_product_matrix,
                reg,
            )
            ipopt = IPOPTSolver(problem)

            self.x0 = ipopt.solve(self.x0)
            self.save_control(self.x0, eta)

    def save_control(self, x0, eta):
        design = os.path.splitext(os.path.basename(self.design_file))[0]
        filename = f"output/{design}/data/N={self.N}_{eta=}.mat"

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        io.savemat(filename, mdict={"data": x0})


if __name__ == "__main__":
    # turn off redundant output in parallel
    df.parameters["std_out_all_processes"] = False

    N = 40
    solver = FluidSolver("designs/twin_pipe.json", N)
    solver.solve()
