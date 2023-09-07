import os

import ufl
import numpy as np
from scipy import io
import dolfin as df
import dolfin_adjoint as dfa

import src.Hs_regularization as Hs_reg
from src.preprocessing import Preprocessing
from src.ipopt_solver import IPOPTSolver, IPOPTProblem
from designs.design_parser import Region, Flow, parse_design

df.set_log_level(df.LogLevel.ERROR)
# turn off redundant output in parallel
df.parameters["std_out_all_processes"] = False


class SidesDomain(df.SubDomain):
    def __init__(self, domain_size: tuple[float, float], sides: list[str]):
        super().__init__()
        self.domain_size = domain_size
        self.sides = sides

    def inside(self, pos, on_boundary):
        if not on_boundary:
            return False

        for side in self.sides:
            if side == "left" and df.near(pos[0], 0.0):
                return True
            elif side == "right" and df.near(pos[0], self.domain_size[0]):
                return True
            elif side == "top" and df.near(pos[1], self.domain_size[1]):
                return True
            elif side == "bottom" and df.near(pos[1], 0.0):
                return True
        return False


class RegionDomain(df.SubDomain):
    def __init__(self, region: Region):
        super().__init__()
        cx, cy = region.center
        w, h = region.size
        self.x_region = (cx - w / 2, cx + w / 2)
        self.y_region = (cy - h / 2, cy + h / 2)

    def inside(self, pos, _):
        return df.between(pos[0], self.x_region) and df.between(pos[1], self.y_region)


class PointDomain(df.SubDomain):
    def __init__(self, point: tuple[float, float]):
        super().__init__()
        self.point = point

    def inside(self, pos, _):
        return df.near(pos[0], self.point[0]) and df.near(pos[1], self.point[1])


class MeshFunctionWrapper:
    """
    A wrapper around 'df.cpp.mesh.MeshFunctionSizet' that handles
    domain indexes for you.
    """

    def __init__(self, mesh: dfa.Mesh):
        self.mesh_function = df.cpp.mesh.MeshFunctionSizet(mesh, 1)
        self.mesh_function.set_all(0)
        self.label_to_idx: dict[str, int] = {}
        self.idx = 1

    def add(self, sub_domain: df.SubDomain, label: str):
        sub_domain.mark(self.mesh_function, self.idx)
        self.label_to_idx[label] = self.idx
        self.idx += 1

    def get(self, label: str):
        return (self.mesh_function, self.label_to_idx[label])


class FlowBC(dfa.UserExpression):
    def __init__(self, **kwargs):
        super().__init__(kwargs)
        self.domain_size: tuple[float, float] = kwargs["domain_size"]
        self.flows: list[Flow] = kwargs["flows"]

    def get_flow(self, position: float, center: float, length: float, rate: float):
        t = position - center
        if -length / 2 < t < length / 2:
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
            elif side == "top" and pos[1] == self.domain_size[1]:
                values[1] -= self.get_flow(pos[0], center, length, rate)
            elif side == "bottom" and pos[1] == 0:
                values[1] += self.get_flow(pos[0], center, length, rate)

    def value_shape(self):
        return (2,)


class FluidSolver:
    def __init__(self, design_file: str, N: int):
        self.design_file = design_file
        self.parameters, flows, no_slip, zero_pressure, max_region = parse_design(
            self.design_file
        )

        if self.parameters.objective == "maximize_flow":
            # why 2 / N? I would understant 1 / N, but why twice that?
            if max_region.size[0] < 2 / N and max_region.size[1] < 2 / N:
                print(
                    "Error: max region is too small. "
                    + f"It's size is ({max_region.size[0]:.10g}, {max_region.size[1]:.10g}), "
                    + f"but either the width or height must be ≥ {2/N}",
                )
                exit(1)

        # define constants
        viscosity = dfa.Constant(1.0)
        self.alpha_max = 2.5 * viscosity / (0.01**2)

        # define domain
        self.N = N
        self.width = self.parameters.width
        self.height = self.parameters.height
        domain_size = (self.width, self.height)

        Nx, Ny = int(self.width * self.N), int(self.height * self.N)

        volume_fraction = self.parameters.fraction
        self.volume = self.width * self.height * volume_fraction
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
        marker = MeshFunctionWrapper(mesh)

        flow_sides = [flow.side for flow in flows]
        marker.add(SidesDomain(domain_size, flow_sides), "flow")

        if zero_pressure:
            marker.add(SidesDomain(domain_size, zero_pressure.sides), "zero_pressure")
        else:
            # default pressure boundary condition: 0 at(0, 0)
            marker.add(PointDomain((0, 0)), "zero_pressure")

        if no_slip:
            marker.add(SidesDomain(domain_size, no_slip.sides), "no_slip")
        else:
            # assume no slip conditions where there is no flow
            all_sides = ["left", "right", "top", "bottom"]
            no_slip_sides = list(set(all_sides).difference(flow_sides))
            marker.add(SidesDomain(domain_size, no_slip_sides), "no_slip")

        if max_region:
            marker.add(RegionDomain(max_region), "max")

        self.boundary_conditions = [
            dfa.DirichletBC(
                self.solution_space.sub(0),
                FlowBC(degree=2, domain_size=domain_size, flows=flows),
                *marker.get("flow"),
            ),
            dfa.DirichletBC(
                self.solution_space.sub(1),
                dfa.Constant(0.0),
                *marker.get("zero_pressure"),
            ),
            dfa.DirichletBC(
                self.solution_space.sub(0),
                dfa.Constant((0.0, 0.0)),
                *marker.get("no_slip"),
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
        (u, _) = df.split(self.forward(rho))

        # objective function
        if self.parameters.objective == "minimize_power":
            main_objective = dfa.assemble(
                0.5 * df.inner(self.alpha(rho) * u, u) * df.dx
                + 0.5 * viscosity * df.inner(df.grad(u), df.grad(u)) * df.dx
            )
        elif self.parameters.objective == "maximize_flow":
            subdomain_data, subdomain_idx = marker.get("max")
            ds = df.Measure("dS", domain=mesh, subdomain_data=subdomain_data)
            main_objective = dfa.assemble(
                df.inner(df.avg(u), dfa.Constant((1.0, 0))) * ds(subdomain_idx)
            )

        # penalty term in objective function
        penalty_objective = dfa.assemble(
            ufl.Max(rho - 1.0, 0.0) ** 2 * df.dx + ufl.Max(-rho - 1.0, 0.0) ** 2 * df.dx
        )

        self.objectives = [main_objective, penalty_objective]
        self.control = dfa.Control(rho)
        self.main_objective_function = dfa.ReducedFunctional(
            main_objective, self.control
        )

        # constraints
        volume_constraint = (
            1.0 / self.volume * dfa.assemble((0.5 * (rho + 1)) * df.dx) - 1.0
        )
        spherical_constraint = dfa.assemble(
            volume_fraction / self.volume * (rho * rho - 1.0) * df.dx
        )
        self.constraints = [
            dfa.ReducedFunctional(volume_constraint, self.control),
            dfa.ReducedFunctional(spherical_constraint, self.control),
        ]

        if self.parameters.objective == "maximize_flow":
            # reference value
            wref = self.forward(dfa.Constant(1.0))
            uref, _ = wref.split(deepcopy=True)
            ref = dfa.assemble(
                0.5 * viscosity * df.inner(df.grad(uref), df.grad(uref)) * df.dx
            )

            dissipation_constraint = (
                dfa.assemble(
                    0.5 * df.inner(self.alpha(rho) * u, u) * df.dx
                    + 0.5 * viscosity * df.inner(df.grad(u), df.grad(u)) * df.dx
                )
                / (130 * ref)
                - 1.0
            )

            self.constraints.append(
                dfa.ReducedFunctional(dissipation_constraint, self.control)
            )

    def alpha(self, rho):
        """Inverse permeability as a function of rho."""
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
        scaling_constraints = [1.0] * len(self.constraints)

        # regularization parameter
        if self.parameters.objective == "maximize_flow":
            reg = 1e-6
        else:
            reg = 10.0

        # different weights for H_sigma matrix
        if self.parameters.objective == "maximize_flow":
            weights = [0.1, 0.01, 0.01, 0.001]
        else:
            weights = [1, 0.01, 0.01, 0.001]

        # different penalization parameters
        etas = [0, 40, 200, 1000]
        if self.parameters.objective == "maximize_flow":
            eta_scale = 1 / 100
        else:
            eta_scale = 1

        # [lower bound, upper bound], for the three constraints (volume, spherical, dissapation)
        initial_bounds = [[0.0, 0.0], [-1.0, 0.0], [-1e6, 0.0]]
        main_bounds = [[-1e6, 0.0], [0.0, 0.0], [-1e6, 0.0]]
        bounds_list = [initial_bounds] + [main_bounds] * (len(etas) - 1)

        for j, (weight, eta, bounds) in enumerate(zip(weights, etas, bounds_list)):
            # we must scale eta to match the magnitude of the objective function
            scaled_eta = eta * eta_scale

            # update inner product
            # consider L2-mass-matrix + weighting * Hs-matrix
            inner_product_matrix = Hs_reg.AssembleHs(
                self.N, self.width / self.height, 7.0 / 16
            ).get_matrix(weight)

            # for performance reasons we first add J and J2 and hand the sum over to the IPOPT solver
            combined_objective = dfa.ReducedFunctional(
                self.objectives[0] + scaled_eta * self.objectives[1], self.control
            )

            if j != 0:
                # move x0 onto sphere
                self.x0 = self.preprocessing.move_onto_sphere(
                    self.x0, self.volume, self.width / self.height
                )

            # solve optimization problem
            problem = IPOPTProblem(
                combined_objective,
                self.constraints,
                scaling_constraints,
                bounds,
                self.preprocessing,
                inner_product_matrix,
                reg,
            )
            ipopt = IPOPTSolver(problem, scaled_eta, j, len(etas))

            self.x0, iterations, objective = ipopt.solve(self.x0)
            self.save_control(self.x0, scaled_eta, iterations, objective)

    def save_control(self, x0, eta, iterations, combined_objective):
        rho = self.preprocessing.dof_to_control(x0)
        main_objective = self.main_objective_function(rho)

        design = os.path.splitext(os.path.basename(self.design_file))[0]
        filename = f"output/{design}/data/N={self.N}_{eta=}.mat"

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        io.savemat(
            filename,
            mdict={
                "data": x0,
                "info": [iterations, main_objective, combined_objective],
            },
        )
