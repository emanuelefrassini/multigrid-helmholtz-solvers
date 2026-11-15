"""Sparse 5-point discretisation; only interior nodes are unknowns.

N is the number of subintervals on each axis; h=1/N and the linear system
has (N-1)^2 unknowns. The physical Dirichlet data enter the right-hand side.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .problem import exact_u, forcing


@dataclass(frozen=True)
class DiscreteProblem:
    N: int
    c: float
    A: sparse.csr_matrix
    rhs: np.ndarray
    exact_interior: np.ndarray

    @property
    def h(self) -> float:
        return 1.0 / self.N

    def full_grid(self, interior: np.ndarray) -> np.ndarray:
        """Return (N+1,N+1) array indexed [y_index,x_index]."""
        x = np.linspace(0, 1, self.N + 1)
        X, Y = np.meshgrid(x, x, indexing="xy")
        u = np.array(exact_u(X, Y), copy=True)
        u[1:-1, 1:-1] = np.asarray(interior).reshape(self.N - 1, self.N - 1)
        return u


def discretize(N: int, c: float = 20.0) -> DiscreteProblem:
    if isinstance(N, bool) or not isinstance(N, (int, np.integer)) or N < 2:
        raise ValueError("N must be an integer >= 2")
    if not np.isfinite(c):
        raise ValueError("c must be finite")
    n = N - 1
    h = 1.0 / N
    T = sparse.diags([-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)],
                     offsets=[-1, 0, 1], format="csr") / h**2
    I = sparse.eye(n, format="csr")
    A = (sparse.kron(I, T, format="csr") + sparse.kron(T, I, format="csr")
         + c * sparse.eye(n*n, format="csr")).tocsr()
    grid = np.arange(1, N, dtype=float) * h
    X, Y = np.meshgrid(grid, grid, indexing="xy")
    b = forcing(X, Y, c).copy()
    # Contributions of boundary neighbours to the interior stencil.
    b[:, 0] += exact_u(0.0, grid) / h**2
    b[:, -1] += exact_u(1.0, grid) / h**2
    b[0, :] += exact_u(grid, 0.0) / h**2
    b[-1, :] += exact_u(grid, 1.0) / h**2
    return DiscreteProblem(N, float(c), A, b.ravel(), exact_u(X, Y).ravel())
