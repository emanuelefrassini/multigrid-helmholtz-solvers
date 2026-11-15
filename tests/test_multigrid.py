import numpy as np
import pytest

from multigrid_helmholtz import (discretize, direct_solve, prolongation_1d,
                                transfer_2d, Multigrid, GaussSeidel,
                                relative_residual, solve_gauss_seidel)


@pytest.mark.parametrize('N', [4,8,16,32])
def test_prolongation_injection_on_shared_coarse_nodes(N):
    p=prolongation_1d(N)
    q=np.arange(1,N//2,dtype=float)
    fine=p@q
    np.testing.assert_allclose(fine[1::2], q)
    assert p.shape == (N-1,N//2-1)


@pytest.mark.parametrize('N', [4,16])
def test_bilinear_and_full_weighting_transfer(N):
    P,R=transfer_2d(N)
    assert P.shape == ((N-1)**2,(N//2-1)**2)
    np.testing.assert_allclose(R.toarray(),.25*P.T.toarray())
    assert np.all(P.data>=0) and np.all(P.data<=1)
    coarse=np.arange((N//2-1)**2,dtype=float)
    out=(P@coarse).reshape(N-1,N-1)
    np.testing.assert_allclose(out[1::2,1::2],coarse.reshape(N//2-1,N//2-1))


def test_galerkin_matrix_is_symmetric_and_spd_for_positive_coefficient():
    d=discretize(16,20)
    mg=Multigrid(d.A,16,levels=3)
    assert len(mg.levels)==3
    for lev in mg.levels:
        np.testing.assert_allclose((lev.A-lev.A.T).toarray(),0,atol=1e-10)
        assert np.linalg.eigvalsh(lev.A.toarray())[0]>0


@pytest.mark.parametrize('N', [16,32,64])
@pytest.mark.parametrize('levels,kind', [(2,'V'),(3,'V'),(3,'W')])
def test_positive_multigrid_converges_to_direct_solution(N,levels,kind):
    d=discretize(N,20.)
    result=Multigrid(d.A,N,levels=levels).solve(d.rhs,kind=kind,tol=1e-8,max_cycles=30)
    assert result.converged
    assert result.iterations < 20
    assert result.residuals[0] == pytest.approx(1.)
    assert result.residuals[-1]<1e-8
    np.testing.assert_allclose(result.solution, direct_solve(d.A,d.rhs),rtol=1e-6,atol=1e-6)


def test_gauss_seidel_sweep_satisfies_lower_triangular_equation():
    from scipy import sparse
    d=discretize(8)
    gs=GaussSeidel(d.A)
    x=np.linspace(0,.2,len(d.rhs))
    updated=gs.sweep(x,d.rhs)
    L=sparse.tril(d.A)
    U=sparse.triu(d.A,1)
    np.testing.assert_allclose(L@updated,d.rhs-U@x,rtol=1e-11,atol=1e-10)


def test_positive_gauss_seidel_matches_direct():
    d=discretize(8,20.)
    result=solve_gauss_seidel(d.A,d.rhs,tol=1e-9,max_cycles=200)
    assert result.converged
    assert result.residuals[-1] < result.residuals[0]
    np.testing.assert_allclose(result.solution,direct_solve(d.A,d.rhs),atol=2e-8)


@pytest.mark.parametrize('N,levels,kind',[(32,2,'V'),(64,3,'V'),(64,3,'W')])
def test_negative_shift_selected_grids_converge(N,levels,kind):
    d=discretize(N,-20.)
    result=Multigrid(d.A,N,levels=levels).solve(d.rhs,kind=kind,tol=1e-8,max_cycles=50)
    assert result.converged
    np.testing.assert_allclose(result.solution,direct_solve(d.A,d.rhs),atol=1e-5)


def test_negative_shift_can_make_a_three_grid_cycle_diverge():
    d=discretize(16,-20.)
    result=Multigrid(d.A,16,levels=3).solve(d.rhs,kind='V',tol=1e-8,max_cycles=25)
    assert not result.converged
    assert result.residuals[-1] > result.residuals[0]


@pytest.mark.parametrize('N,levels',[(15,2),(10,3),(8,1),(4,4)])
def test_invalid_hierarchy_is_rejected(N,levels):
    d=discretize(N)
    with pytest.raises(ValueError):
        Multigrid(d.A,N,levels=levels)


def test_bad_cycle_and_tolerance_are_rejected():
    d=discretize(8)
    mg=Multigrid(d.A,8,levels=2)
    with pytest.raises(ValueError):mg.cycle(d.rhs,kind='X')
    with pytest.raises(ValueError):mg.solve(d.rhs,tol=0)
    with pytest.raises(ValueError):solve_gauss_seidel(d.A,d.rhs,max_cycles=-1)
