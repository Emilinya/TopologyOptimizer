import dolfin as df
import dolfin_adjoint as dfa
import numpy as np
from scipy import io
import ufl

df.set_log_level(df.LogLevel.ERROR)

import Hs_regularization as Hs_reg
from ipopt_solver import IPOPTSolver, IPOPTProblem
from preprocessing import Preprocessing

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
class InflowOutflow(dfa.UserExpression):
    def __init__(self, **kwargs):
        super(InflowOutflow, self).__init__(kwargs)
        self.domain_size = kwargs["domain_size"]

    def eval(self, values, x):
        values[1] = 0.0
        values[0] = 0.0
        l = 1.0 / 6.0
        gbar = 1.0

        if x[0] == 0.0 or x[0] == self.domain_size[0]:
            if (1.0 / 4 - l / 2) < x[1] < (1.0 / 4 + l / 2):
                t = x[1] - 1.0 / 4
                values[0] = gbar * (1 - (2 * t / l) ** 2)
            if (3.0 / 4 - l / 2) < x[1] < (3.0 / 4 + l / 2):
                t = x[1] - 3.0 / 4
                values[0] = gbar * (1 - (2 * t / l) ** 2)

    def value_shape(self):
        return (2,)


# Define the boundary condition on pressure
class PressureB(df.SubDomain):
    def inside(self, x, on_boundary):
        return df.near(x[0], (0.0)) and df.near(x[1], (0.0))


def alpha(rho):
    """Inverse permeability as a function of rho, equation (40)"""
    equation = alpha_max * (
        -1.0 / 16 * rho**4 + 3.0 / 8 * rho**2 - 0.5 * rho + 3.0 / 16
    )
    return df.conditional(
        df.gt(rho, 1.0),
        0.0,
        df.conditional(df.gt(rho, -1.0), equation, -1.0 * alpha_max * rho),
    )


def forward(rho, flowBC, pressureBC):
    """Solve the forward problem for a given fluid distribution rho(x)."""
    w = dfa.Function(solution_space)
    (u, p) = df.TrialFunctions(solution_space)
    (v, q) = df.TestFunctions(solution_space)

    F = (
        alpha(rho) * df.inner(u, v)
        + df.inner(df.grad(u), df.grad(v))
        + df.inner(df.grad(p), v)
        + df.inner(df.div(u), q)
    ) * df.dx
    bc = [
        dfa.DirichletBC(solution_space.sub(0), flowBC, "on_boundary"),
        dfa.DirichletBC(
            solution_space.sub(1), dfa.Constant(0.0), pressureBC, method="pointwise"
        ),
    ]
    dfa.solve(df.lhs(F) == df.rhs(F), w, bcs=bc)

    return w


def save_control(x0, eta, params):
    filename = f"output/data/design_{N=}_{eta=}.mat"
    io.savemat(filename, mdict={"data": x0, "w": params["w"], "h": params["h"]})


if __name__ == "__main__":
    # turn off redundant output in parallel
    df.parameters["std_out_all_processes"] = False

    # define constatns
    viscosity = dfa.Constant(1.0)
    alpha_max = 2.5 * viscosity / (0.01**2)

    # define domain
    N = 40
    width = 1.5
    height = 1
    volume_fraction = 1 / 3
    V = width * height * volume_fraction
    mesh = dfa.Mesh(
        dfa.RectangleMesh(
            df.MPI.comm_world,
            df.Point(0.0, 0.0),
            df.Point(width, height),
            int(width * N),
            int(height * N),
        )
    )

    # define solution space
    control_space = df.FunctionSpace(mesh, "DG", 0)
    velocity_space = df.VectorElement("CG", mesh.ufl_cell(), 2)
    pressure_space = df.FiniteElement("CG", mesh.ufl_cell(), 1)
    solution_space = df.FunctionSpace(mesh, velocity_space * pressure_space)

    # define boundary conditions
    flowBC = InflowOutflow(degree=2, domain_size=(width, height))
    pressureBC = PressureB()

    # create initial conditions
    k = int(width * N) * int(height * N) * 2
    x0 = (2.0 * volume_fraction - 1) * np.ones(int(k / 2))

    # preprocessing class which contains dof_to_control-mapping
    preprocessing = Preprocessing(N, control_space)
    rho = preprocessing.dof_to_control(x0)

    # get reduced objective function: rho --> j(rho)
    dfa.set_working_tape(dfa.Tape())
    w = forward(rho, flowBC, pressureBC)
    (u, p) = df.split(w)

    # objective function
    J = dfa.assemble(
        0.5 * df.inner(alpha(rho) * u, u) * df.dx
        + 0.5 * viscosity * df.inner(df.grad(u), df.grad(u)) * df.dx
    )
    # penalty term in objective function
    J2 = dfa.assemble(
        ufl.Max(rho - 1.0, 0.0) ** 2 * df.dx + ufl.Max(-rho - 1.0, 0.0) ** 2 * df.dx
    )

    Js = [J, J2]
    m = dfa.Control(rho)
    Jeval = dfa.ReducedFunctional(J, m)
    # Note: the evaluation of the gradient can be speed up since the adjoint solve requires no pde solve
    # (see Appendix A.4)

    # constraints
    volume_constraint = 1.0 / V * dfa.assemble((0.5 * (rho + 1)) * df.dx) - 1.0
    spherical_constraint = dfa.assemble(volume_fraction / V * (rho * rho - 1.0) * df.dx)
    constraints = [
        dfa.ReducedFunctional(volume_constraint, m),
        dfa.ReducedFunctional(spherical_constraint, m),
    ]
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
        inner_product_matrix = Hs_reg.AssembleHs(N, width, sigma).get_matrix(weight)

        # for performance reasons we first add J and J2 and hand the sum over to the IPOPT solver
        J_ = Js[0] + eta * Js[1]
        Jhat = [dfa.ReducedFunctional(J_, m)]
        # Note: the evaluation of the gradient can be speed up since the adjoint solve requires no pde solve
        # (see Appendix A.4)

        if j != 0:
            # move x0 onto sphere
            x0 = preprocessing.move_onto_sphere(x0, V, width)

        # solve optimization problem
        problem = IPOPTProblem(
            Jhat,
            [1.0, eta],
            constraints,
            scaling_constraints,
            bound,
            preprocessing,
            inner_product_matrix,
            reg,
        )
        ipopt = IPOPTSolver(problem)

        x0 = ipopt.solve(x0)
        save_control(x0, eta, {"w": width, "h": height})
