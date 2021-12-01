from dolfin import *
from dolfin_adjoint import *
from copy import deepcopy
import numpy as np
import backend

from pyadjoint.optimization.optimization_solver import OptimizationSolver
from pyadjoint.reduced_functional_numpy import ReducedFunctionalNumPy


import ipopt

import matplotlib.pyplot as plt


class IPOPTSolver(OptimizationSolver):
    def __init__(self, preprocessing, num_dofs_DG0_quad, rf, param, h, V, parameters=None):
        try:
            import ipopt
        except ImportError:
            print("You need to install cyipopt. (It is recommended to install IPOPT with HSL support!)")
            raise
        self.preprocessing = preprocessing
        self.k = int(num_dofs_DG0_quad/2)
        self.h = h
        self.V = V
        self.param = param
        self.rfn = ReducedFunctionalNumPy(rf)
        self.problem_obj = self.create_problem_obj(self)

        print('Initialization of IPOPTSolver finished')

    def create_problem_obj(self, outer):
        return IPOPTSolver.shape_opt_prob(outer)

    def test_objective(self):
        # check dof_to_deformation with first order derivative check
        print('Extension.test_dof_to_deformation started.......................')
        xl = self.k
        x0 = -0.5 * np.ones(xl) # 0.5 * np.ones(xl)
        ds = 1.0 * np.ones(xl)
        # ds = interpolate(Expression('0.2*x[0]', degree=1), self.Vd)
        j0 = self.problem_obj.objective(x0)
        djx = self.problem_obj.gradient(x0)
        epslist = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 1e-5, 1e-6, 1e-7]
        jlist = [self.problem_obj.objective(x0 + eps * ds) for eps in epslist]
        self.perform_first_order_check(jlist, j0, djx, ds, epslist)
        return

    def test_constraints(self):
        # check dof_to_deformation with first order derivative check
        print('Extension.test_dof_to_deformation started.......................')
        xl = self.k
        x0 = 0.5 * np.ones(xl)
        j0 = self.problem_obj.constraints(x0)
        djx = self.problem_obj.jacobian(x0)
        ds = 1.0 * np.ones(xl)
        # ds = interpolate(Expression('0.2*x[0]', degree=1), self.Vd)
        j0 = self.problem_obj.constraints(x0)
        djx = self.problem_obj.jacobian(x0)
        epslist = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 1e-5, 1e-6, 1e-7]
        jlist = [self.problem_obj.constraints(x0 + eps * ds) for eps in epslist]
        self.perform_first_order_check(jlist, j0, djx, ds, epslist)
        return

    def perform_first_order_check(self, jlist, j0, gradj0, ds, epslist):
        # j0: function value at x0
        # gradj0: gradient value at x0
        # epslist: list of decreasing eps-values
        # jlist: list of function values at x0+eps*ds for all eps in epslist
        diff0 = []
        diff1 = []
        order0 = []
        order1 = []
        i = 0
        for eps in epslist:
            je = jlist[i]
            di0 = je - j0
            di1 = je - j0 - eps * np.dot(gradj0, ds)
            diff0.append(abs(di0))
            diff1.append(abs(di1))
            if i == 0:
                order0.append(0.0)
                order1.append(0.0)
            if i > 0:
                order0.append(np.log(diff0[i - 1] / diff0[i]) / np.log(epslist[i - 1] / epslist[i]))
                order1.append(np.log(diff1[i - 1] / diff1[i]) / np.log(epslist[i - 1] / epslist[i]))
            i = i + 1
        for i in range(len(epslist)):
            print('eps\t', epslist[i], '\t\t check continuity\t', order0[i], '\t\t diff0 \t', diff0[i],
                  '\t\t check derivative \t', order1[i], '\t\t diff1 \t', diff1[i], '\n'),

        return

    class shape_opt_prob(object):
        def __init__(self, outer):
            self.preprocessing = outer.preprocessing
            self.rfn = outer.rfn
            self.param = outer.param
            self.h = outer.h
            self.V = outer.V

        def objective(self, x):
            #
            # The callback for calculating the objective
            #
            # x to deformation
            print('evaluate objective')
            transformed_x = self.preprocessing.transformation(x)
            rho = self.preprocessing.dof_to_rho(transformed_x)
            j1 = self.rfn(rho.vector()[:])
            print('j1', j1)
            j = (j1+ 0.5 * self.param["reg"] * np.dot(np.asarray(x), np.asarray(x))
                 + 0.5 * self.param["penal"]* np.dot(np.maximum(-1.0*np.ones(len(x)) - np.asarray(x), np.zeros(len(x))),
                                                    np.maximum(-1.0*np.ones(len(x)) -np.asarray(x), np.zeros(len(x))))
                 + 0.5 * self.param["penal"] * np.dot(
                        np.maximum(np.asarray(x) - 1.0*np.ones(len(x)), np.zeros(len(x))),
                        np.maximum(np.asarray(x) - 1.0*np.ones(len(x)), np.zeros(len(x))))
                 )
            return j

        def gradient(self, x):
            #
            # The callback for calculating the gradient
            #
            # print('evaluate derivative of objective funtion')
            print('evaluate gradient')
            transformed_x = self.preprocessing.transformation(x)
            rho = self.preprocessing.dof_to_rho(transformed_x)
            new_params = [self.__copy_data(p.data()) for p in self.rfn.controls]
            self.rfn.set_local(new_params, rho.vector().get_local())
            dJf = self.rfn.derivative(forget=False, project=False)  # rf
            dJ = self.preprocessing.dof_to_rho_chainrule(dJf, 2)
            dJ = self.preprocessing.transformation_chainrule(dJ)
            dJ = dJ + self.param["reg"] * x + self.param["penal"]*self.h*self.h*(
                np.maximum(np.asarray(x) - 1.0 * np.ones(len(x)), np.zeros(len(x))) -
                np.maximum(-1.0 * np.ones(len(x)) - np.asarray(x), np.zeros(len(x)))
            )
            return np.asarray(dJ, dtype=float)

        def constraints(self, x):
            #
            # The callback for calculating the constraints
            print('evaluate constraint')
            scale2 = 1.0/(self.h*self.h*np.dot(np.ones(len(x)),np.ones(len(x))))
            s = self.param["sphere"]*scale2*self.h*self.h\
                *(np.dot(np.asarray(x), np.asarray(x))-np.dot(np.ones(len(x)),np.ones(len(x))))
            transformed_x = self.preprocessing.transformation(x)
            rho = self.preprocessing.dof_to_rho(transformed_x)
            v = scale2*(self.param["vol"]*assemble((0.5*(rho+1))*dx)-self.V)
            return np.array((s,v))  # , d_ct))

        def jacobian(self, x):
            #
            # The callback for calculating the Jacobian
            #
            print('evaluate jacobian')
            scale2 = 1.0 /(self.h * self.h * np.dot(np.ones(len(x)), np.ones(len(x))))
            ds = scale2*self.param["sphere"]*2*self.h*self.h*np.asarray(x)
            transformed_x = self.preprocessing.transformation(x)
            rho = self.preprocessing.dof_to_rho(transformed_x)
            psiv = TestFunction(rho.function_space())
            dv = assemble((0.5*psiv*dx))
            dv = self.preprocessing.dof_to_rho_chainrule(dv, 2)
            dv = scale2*self.param["vol"]*self.preprocessing.transformation_chainrule(dv)
            return np.concatenate((ds,dv))  # , d_ct_d))

        # def hessianstructure(self):
        #    #
        #    # The structure of the Hessian
        #    # Note:
        #    # The default hessian structure is of a lower triangular matrix. Therefore
        #    # this function is redundant. I include it as an example for structure
        #    # callback.
        #    #
        #    global hs
        #
        #    hs = sps.coo_matrix(np.tril(np.ones((4, 4))))
        #    return (hs.col, hs.row)
        #
        # def hessian(self, x, lagrange, obj_factor):
        #    #
        #    # The callback for calculating the Hessian
        #    #
        #    H = obj_factor*np.array((
        #            (2*x[3], 0, 0, 0),
        #            (x[3],   0, 0, 0),
        #            (x[3],   0, 0, 0),
        #            (2*x[0]+x[1]+x[2], x[0], x[0], 0)))
        #
        #    H += lagrange[0]*np.array((
        #            (0, 0, 0, 0),
        #            (x[2]*x[3], 0, 0, 0),
        #            (x[1]*x[3], x[0]*x[3], 0, 0),
        #            (x[1]*x[2], x[0]*x[2], x[0]*x[1], 0)))
        #
        #    H += lagrange[1]*2*np.eye(4)
        #
        #    #
        #    # Note:
        #    #
        #    #
        #    return H[hs.row, hs.col]

        def intermediate(
                self,
                alg_mod,
                iter_count,
                obj_value,
                inf_pr,
                inf_du,
                mu,
                d_norm,
                regularization_size,
                alpha_du,
                alpha_pr,
                ls_trials
        ):

            #
            # Example for the use of the intermediate callback.
            #
            print("Objective value at iteration ", iter_count, " is ", obj_value)
            return

        def __copy_data(self, m):
            """Returns a deep copy of the given Function/Constant."""
            if hasattr(m, "vector"):
                return backend.Function(m.function_space())
            elif hasattr(m, "value_size"):
                return backend.Constant(m(()))
            else:
                raise TypeError('Unknown control type %s.' % str(type(m)))

    def solve(self, x0):
        max_float = np.finfo(np.double).max
        min_float = np.finfo(np.double).min

        c_vol = self.param["relax_vol"]
        c_sph = self.param["relax_sphere"]
        w_v = self.param["vol"]
        w_s = self.param["sphere"]
        cl = [w_s*c_sph[0], w_v*c_vol[0]]
        cu = [w_s*c_sph[1], w_v*c_vol[1]]

        ub = np.array([max_float] * len(x0))
        lb = np.array([min_float] * len(x0))

        nlp = ipopt.problem(
            n=len(x0),
            m=len(cl),
            problem_obj=self.problem_obj,
            lb=lb,
            ub=ub,
            cl=cl,
            cu=cu
        )

        nlp.addOption('mu_strategy', 'adaptive')
        # nlp.addOption('derivative_test', 'first-order')
        nlp.addOption('point_perturbation_radius', 0.0)
        nlp.addOption('max_iter', self.param["maxiter_IPOPT"])
        nlp.addOption('tol', 1e-3)

        x, info = nlp.solve(x0)
        return x