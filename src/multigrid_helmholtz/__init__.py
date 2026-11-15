"""Numerical methods for a 2-D manufactured Helmholtz-type problem."""
from .discretization import DiscreteProblem, discretize
from .problem import exact_u, forcing
from .solvers import (GaussSeidel, Multigrid, SolverResult, direct_solve,
                      relative_residual, solve_gauss_seidel)
from .transfer import prolongation_1d, transfer_2d

__all__ = ["DiscreteProblem", "discretize", "exact_u", "forcing", "GaussSeidel",
           "Multigrid", "SolverResult", "direct_solve", "relative_residual",
           "solve_gauss_seidel", "prolongation_1d", "transfer_2d"]
