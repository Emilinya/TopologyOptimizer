from dolfin import *
from dolfin_adjoint import *
import numpy as np
from scipy import io
import ufl

set_log_level(LogLevel.ERROR)

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
class InflowOutflow(UserExpression):
    def init(self, delta):
        self.delta = delta

    def eval(self, values, x):
        if not hasattr(self, "delta"):
            print("Tried to eval InflowOutflow before calling init!")
            exit(1)

        values[1] = 0.0
        values[0] = 0.0
        l = 1.0 / 6.0
        gbar = 1.0

        if x[0] == 0.0 or x[0] == self.delta:
            if (1.0 / 4 - l / 2) < x[1] < (1.0 / 4 + l / 2):
                t = x[1] - 1.0 / 4
                values[0] = gbar * (1 - (2 * t / l) ** 2)
            if (3.0 / 4 - l / 2) < x[1] < (3.0 / 4 + l / 2):
                t = x[1] - 3.0 / 4
                values[0] = gbar * (1 - (2 * t / l) ** 2)

    def value_shape(self):
        return (2,)


# Define the boundary condition on pressure
class PressureB(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], (0.0)) and near(x[1], (0.0))


def alpha(rho):
    """Inverse permeability as a function of rho, equation (40)"""
    equation = alphabar * (
        -1.0 / 16 * rho**4 + 3.0 / 8 * rho**2 - 0.5 * rho + 3.0 / 16
    )
    return conditional(
        gt(rho, 1.0), 0.0, conditional(gt(rho, -1.0), equation, -1.0 * alphabar * rho)
    )


def forward(rho, flowBC, pressureBC):
    """Solve the forward problem for a given fluid distribution rho(x)."""
    w = Function(W)
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    F = (
        alpha(rho) * inner(u, v) * dx
        + inner(grad(u), grad(v)) * dx
        + inner(grad(p), v) * dx
        + inner(div(u), q) * dx
    )
    bc = [
        DirichletBC(W.sub(0), flowBC, "on_boundary"),
        DirichletBC(W.sub(1), Constant(0.0), pressureBC, method="pointwise"),
    ]
    solve(lhs(F) == rhs(F), w, bcs=bc)

    return w


def save_control(x0, controls_file, eta, params, J=None):
    rho = preprocessing.dof_to_control(x0)
    rho.rename("density", "density")
    print("objective function value J", J(rho))
    controls_file << rho

    filename = "output/data/design_" + f"{N=}_{eta=}" + ".mat"
    io.savemat(filename, mdict={"data": x0, "w": params["w"], "h": params["h"]})


if __name__ == "__main__":
    # turn off redundant output in parallel
    parameters["std_out_all_processes"] = False

    mu = Constant(1.0)  # viscosity
    alphabar = 2.5 * mu / (0.01**2)  # parameter for \alpha

    params = {
        "N": 40,
        "w": 1.5,
        "h": 1.0,
    }
    N, delta = params["N"], params["w"]

    V = (1.0 * delta) / 3  # want the fluid to occupy 1/3 of the domain
    mesh = Mesh(
        RectangleMesh(
            MPI.comm_world, Point(0.0, 0.0), Point(delta, 1.0), int(delta * N), N
        )
    )

    controls_file = File("output/extra/final_controls_" + str(N) + ".pvd")

    U_h = VectorElement("CG", mesh.ufl_cell(), 2)
    P_h = FiniteElement("CG", mesh.ufl_cell(), 1)
    W = FunctionSpace(mesh, U_h * P_h)  # mixed Taylor-Hood function space

    B = FunctionSpace(mesh, "DG", 0)  # control function space
    b = Function(B)
    k = len(b.vector()[:])
    b.vector()[:] = range(k)

    flowBC = InflowOutflow(degree=2)
    flowBC.init(delta)
    pressureBC = PressureB()

    x0 = (2.0 * V / delta - 1) * np.ones(int(k / 2))

    # preprocessing class which contains dof_to_control-mapping
    sigma = 7.0 / 16
    preprocessing = Preprocessing(N, B)

    rho = preprocessing.dof_to_control(x0)

    # get reduced objective function: rho --> j(rho)
    set_working_tape(Tape())
    w = forward(rho, flowBC, pressureBC)
    (u, p) = split(w)

    controls = File("output/extra/control_iterations_guess" + str(N) + ".pvd")
    allctrls = File("output/extra/allcontrols_" + str(N) + ".pvd")
    rho_viz = Function(B, name="ControlVisualisation")

    def eval_cb(j, rho):
        rho_viz.assign(rho)
        controls << rho_viz
        allctrls << rho_viz

    # objective function
    J = assemble(
        0.5 * inner(alpha(rho) * u, u) * dx + 0.5 * mu * inner(grad(u), grad(u)) * dx
    )
    # penalty term in objective function
    J2 = assemble(
        ufl.Max(rho - 1.0, 0.0) ** 2 * dx + ufl.Max(-rho - 1.0, 0.0) ** 2 * dx
    )
    #
    Js = [J, J2]
    m = Control(rho)

    Jeval = ReducedFunctional(J, m)
    # Note: the evaluation of the gradient can be speed up since the adjoint solve requires no pde solve
    # (see Appendix A.4)

    # constraints
    v = 1.0 / V * assemble((0.5 * (rho + 1)) * dx) - 1.0  # volume constraint
    s = assemble(1.0 / delta * (rho * rho - 1.0) * dx)  # spherical constraint
    constraints = [ReducedFunctional(v, m), ReducedFunctional(s, m)]
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
        inner_product_matrix = Hs_reg.AssembleHs(N, delta, sigma).get_matrix(weight)

        scaling_Js = [1.0, eta]

        # for performance reasons we first add J and J2 and hand the sum over to the IPOPT solver
        J_ = 0
        for i in range(len(Js)):
            J_ += Js[i] * scaling_Js[i]
        Jhat = [ReducedFunctional(J_, m, eval_cb_post=eval_cb)]
        # Note: the evaluation of the gradient can be speed up since the adjoint solve requires no pde solve
        # (see Appendix A.4)

        if j != 0:
            # move x0 onto sphere
            x0 = preprocessing.move_onto_sphere(x0, V, delta)

        # solve optimization problem
        problem = IPOPTProblem(
            Jhat,
            scaling_Js,
            constraints,
            scaling_constraints,
            bound,
            preprocessing,
            inner_product_matrix,
            reg,
        )
        ipopt = IPOPTSolver(problem)

        x0 = ipopt.solve(x0)
        save_control(x0, controls_file, eta, params, J=Jeval)
