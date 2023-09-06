from dolfin import *
from dolfin_adjoint import *

from src.ipopt_solver import IPOPTProblem, IPOPTSolver
from src.preprocessing import Preprocessing
import src.Hs_regularization as Hs_reg
from test_preprocessing import setting
import numpy as np
import pytest

def test_ipopt_cholesky():
    # load Hs matrix
    N = 10
    delta = 1.0
    sigma = 7. / 16
    reg_Hs = Hs_reg.AssembleHs(N, delta, sigma)
    matrix = reg_Hs.get_matrix(1.0)

    # apply cholesky
    U = IPOPTProblem.sparse_cholesky(matrix)

    # check if U^T U = matrix
    UTU = U.transpose().dot(U)
    assert np.all(matrix.todense() - UTU.todense() < 1e-13)

def test_objective():
    # setting
    N = 10
    delta = 1.0
    sigma = 7. / 16
    reg_Hs = Hs_reg.AssembleHs(N, delta, sigma)
    matrix = reg_Hs.get_matrix(1.0)
    mesh, B, k = setting(N)
    preproc = Preprocessing(N, B)

    # reduced objective
    tape = Tape()
    set_working_tape(tape)
    b = Function(B)
    J = assemble(inner(b,b)*dx(mesh))
    Jhat = ReducedFunctional(J, Control(b))

    # initialize IPOPT
    problem = IPOPTProblem(Jhat, [None], None, None, preproc, matrix, 2.0)

    # test objective
    order1, diff1 = IPOPTSolver(problem, 0, 0, 1).test_objective(int(k/2))

    assert order1[-1] > 1.8

def constraints_setting():
    # setting
    N = 10
    delta = 1.0
    sigma = 7. / 16
    reg_Hs = Hs_reg.AssembleHs(N, delta, sigma)
    matrix = reg_Hs.get_matrix(1.0)
    mesh, B, k = setting(N)
    preproc = Preprocessing(N, B)

    # constraints
    tape = Tape()
    set_working_tape(tape)
    b = Function(B)
    J1 = assemble((b -1)*dx)
    J2 = assemble((b*b -2)*dx)
    con1 = ReducedFunctional(J1, Control(b))
    con2 = ReducedFunctional(J2, Control(b))
    constraints = [con1, con2]
    scaling = [1.0, 1.0]
    bounds = [[-10, 10], [-1, 1]]

    # initialize IPOPT
    problem = IPOPTProblem(None, constraints, scaling, bounds, preproc, matrix, 2.0)

    return problem, k

@pytest.mark.parametrize(
    "ind", [0,1]
)
def test_constraints(ind):
    problem, k = constraints_setting()
    # test constraints
    order1, diff1 = IPOPTSolver(problem, 0, 0, 1).test_constraints(int(k/2), ind, option=1)
    assert order1[-1] > 1.8 or diff1[-1] < 1e-12
