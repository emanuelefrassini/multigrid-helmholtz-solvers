import numpy as np
import pytest

from multigrid_helmholtz import discretize, direct_solve, exact_u, forcing, relative_residual


def test_exact_solution_boundary_values():
    for x in np.linspace(0, 1, 8):
        assert exact_u(x, 0.0) == pytest.approx(np.exp(x))
        assert exact_u(x, 1.0) == pytest.approx(np.exp(x))
    for y in np.linspace(0, 1, 8):
        assert exact_u(0.0, y) == pytest.approx(1.0)
        assert exact_u(1.0, y) == pytest.approx(np.e)


@pytest.mark.parametrize('c', [20.0, -20.0])
def test_forcing_matches_finite_difference_derivative(c):
    x, y = 0.39, 0.61
    h = 2e-4
    lap = ((exact_u(x+h,y)-2*exact_u(x,y)+exact_u(x-h,y))/h**2
           + (exact_u(x,y+h)-2*exact_u(x,y)+exact_u(x,y-h))/h**2)
    assert forcing(x,y,c) == pytest.approx(-lap+c*exact_u(x,y), abs=2e-6)


@pytest.mark.parametrize('c', [20.0, -20.0])
def test_matrix_is_symmetric_and_rhs_has_correct_size(c):
    p = discretize(8, c)
    assert p.A.shape == (49,49)
    assert p.A.nnz == 5*49-4*7
    np.testing.assert_allclose((p.A-p.A.T).toarray(), 0.0)
    assert p.rhs.shape == (49,)
    assert p.full_grid(p.exact_interior).shape == (9,9)


def test_stencil_on_center_node():
    p = discretize(8,20)
    row = p.A.getrow(24).toarray().ravel()
    assert row[24] == pytest.approx(4*64+20)
    for j in [17,23,25,31]:
        assert row[j] == pytest.approx(-64)
    assert np.count_nonzero(row)==5


@pytest.mark.parametrize('c', [20., -20.])
def test_direct_residual_and_second_order_accuracy(c):
    errors = []
    for N in [8,16,32]:
        p=discretize(N,c)
        x=direct_solve(p.A,p.rhs)
        assert relative_residual(p.A,p.rhs,x)<1e-11
        errors.append(np.max(np.abs(x-p.exact_interior)))
        full=p.full_grid(x)
        expected=p.full_grid(p.exact_interior)
        assert np.max(np.abs(full[0]-expected[0]))<1e-14
        assert np.max(np.abs(full[-1]-expected[-1]))<1e-14
        assert np.max(np.abs(full[:,0]-expected[:,0]))<1e-14
        assert np.max(np.abs(full[:,-1]-expected[:,-1]))<1e-14
    assert errors[2] < 0.45*errors[1]
    assert errors[1] < 0.6*errors[0]


@pytest.mark.parametrize('bad', [0,1,-1,5.2,True])
def test_bad_grid_rejected(bad):
    with pytest.raises(ValueError):
        discretize(bad)
