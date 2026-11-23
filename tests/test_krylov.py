"""Data-free numerical tests for shifted-multigrid GMRES."""
import numpy as np
import pytest

from multigrid_helmholtz import Multigrid, direct_solve, discretize, relative_residual
from multigrid_helmholtz.krylov import ShiftedMultigridPreconditioner, gmres_solve


@pytest.mark.parametrize("N", [16, 32, 64])
def test_shifted_preconditioned_gmres_solves_indefinite_problem(N):
    p = discretize(N, -20.0)
    M = ShiftedMultigridPreconditioner(p.A, N, reaction=-20.0, surrogate_reaction=20.0)
    result = gmres_solve(p.A, p.rhs, preconditioner=M, tol=1e-8, maxiter=20)
    assert result.converged and result.info == 0
    assert result.true_relative_residual < 1e-8
    assert result.iterations > 0
    assert len(result.preconditioned_residual_history) == result.iterations
    np.testing.assert_allclose(result.solution, direct_solve(p.A, p.rhs), rtol=2e-5, atol=2e-5)


def test_unshifted_multigrid_diverges_but_shifted_krylov_converges():
    p = discretize(16, -20.0)
    legacy = Multigrid(p.A, 16, levels=3).solve(p.rhs, max_cycles=25)
    assert not legacy.converged
    M = ShiftedMultigridPreconditioner(p.A, 16, -20.0)
    improved = gmres_solve(p.A, p.rhs, preconditioner=M, tol=1e-8)
    assert improved.converged


def test_surrogate_matrix_equals_original_plus_real_shift():
    p = discretize(16, -20.0)
    M = ShiftedMultigridPreconditioner(p.A, 16, -20.0, surrogate_reaction=20.0)
    np.testing.assert_allclose((M.surrogate_matrix - p.A).diagonal(), 40.0)
    np.testing.assert_allclose((M.surrogate_matrix - discretize(16, 20.0).A).toarray(), 0., atol=1e-12)
    assert relative_residual(M.surrogate_matrix, np.ones(len(p.rhs)), direct_solve(M.surrogate_matrix, np.ones(len(p.rhs)))) < 1e-12


@pytest.mark.parametrize("cycle", ["V", "W"])
def test_preconditioner_is_linear_and_zero_started(cycle):
    p = discretize(16, -20.0)
    M = ShiftedMultigridPreconditioner(p.A, 16, -20.0, cycle=cycle)
    rng = np.random.default_rng(8)
    u, v = rng.normal(size=(2, len(p.rhs)))
    np.testing.assert_allclose(M.apply(2 * u - 3 * v), 2 * M.apply(u) - 3 * M.apply(v), rtol=1e-11, atol=1e-12)
    np.testing.assert_array_equal(M.apply(np.zeros_like(u)), np.zeros_like(u))
    np.testing.assert_allclose(M.operator @ u, M.apply(u))


def test_plain_gmres_on_small_indefinite_grid():
    p = discretize(16, -20.0)
    out = gmres_solve(p.A, p.rhs, restart=30, maxiter=20)
    assert out.converged and out.info == 0
    assert out.true_relative_residual < 1e-8


def test_fixed_preconditioner_reproducibility():
    p = discretize(16, -20.0)
    M = ShiftedMultigridPreconditioner(p.A, 16, -20.0)
    a = gmres_solve(p.A, p.rhs, preconditioner=M)
    b = gmres_solve(p.A, p.rhs, preconditioner=M)
    np.testing.assert_array_equal(a.solution, b.solution)
    np.testing.assert_array_equal(a.preconditioned_residual_history, b.preconditioned_residual_history)


def test_krylov_with_positive_reaction_coefficient():
    p = discretize(16, 20.0)
    M = ShiftedMultigridPreconditioner(p.A, 16, 20.0, surrogate_reaction=20.0)
    a = gmres_solve(p.A, p.rhs, preconditioner=M)
    assert a.converged and a.true_relative_residual < 1e-8


def test_zero_rhs_and_exact_initial_guess():
    p = discretize(16, -20.0)
    M = ShiftedMultigridPreconditioner(p.A, 16, -20.0)
    z = gmres_solve(p.A, np.zeros(len(p.rhs)), preconditioner=M)
    assert z.converged and z.iterations == 0
    np.testing.assert_allclose(z.solution, 0., atol=1e-15)
    sol = direct_solve(p.A, p.rhs)
    exact = gmres_solve(p.A, p.rhs, x0=sol, preconditioner=M)
    assert exact.converged
    assert exact.true_relative_residual < 1e-8


def test_iteration_limit_is_reported_without_claiming_convergence():
    p = discretize(32, -20.0)
    out = gmres_solve(p.A, p.rhs, tol=1e-12, restart=1, maxiter=1)
    assert not out.converged
    assert out.info != 0 and out.iterations == 1
    assert out.true_relative_residual > 1e-12


@pytest.mark.parametrize("kwargs", [
    dict(tol=0), dict(tol=float("nan")), dict(restart=0),
    dict(restart=True), dict(maxiter=0), dict(maxiter=True),
    dict(x0=np.zeros(2)),
])
def test_invalid_gmres_arguments(kwargs):
    p = discretize(8)
    with pytest.raises(ValueError):
        gmres_solve(p.A, p.rhs, **kwargs)


@pytest.mark.parametrize("kwargs", [
    dict(surrogate_reaction=-10), dict(reaction=float('nan')),
    dict(surrogate_reaction=float('inf')), dict(cycle='X'),
    dict(levels=1),
])
def test_invalid_preconditioner_arguments(kwargs):
    p = discretize(8, -20)
    with pytest.raises(ValueError):
        ShiftedMultigridPreconditioner(p.A, 8, **({"reaction": -20} | kwargs))


def test_inconsistent_problem_dimensions_are_rejected():
    p = discretize(8)
    with pytest.raises(ValueError):
        ShiftedMultigridPreconditioner(p.A, 16, 20)
    with pytest.raises(ValueError):
        gmres_solve(p.A, np.ones(9))
    with pytest.raises(ValueError):
        ShiftedMultigridPreconditioner(p.A, 8, 20).apply(np.ones(9))
    with pytest.raises(ValueError):
        gmres_solve(p.A, p.rhs, preconditioner=ShiftedMultigridPreconditioner(discretize(16).A, 16, 20))
