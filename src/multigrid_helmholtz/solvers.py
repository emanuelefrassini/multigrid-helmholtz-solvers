"""Sparse direct solve, forward Gauss-Seidel, and Galerkin multigrid.

Multigrid uses correction cycles; GS is a triangular solve, never a dense
iteration matrix. V cycles use gamma=1, W cycles gamma=2.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, spsolve_triangular, splu

from .transfer import transfer_2d


@dataclass
class SolverResult:
    solution: np.ndarray
    residuals: np.ndarray  # includes initial relative residual at index 0
    converged: bool

    @property
    def iterations(self) -> int:
        return len(self.residuals) - 1


def relative_residual(A: sparse.spmatrix, b: np.ndarray,
                      x: np.ndarray) -> float:
    scale = np.linalg.norm(b)
    return float(np.linalg.norm(b - A @ x) / (scale if scale else 1.0))


def direct_solve(A: sparse.spmatrix, b: np.ndarray) -> np.ndarray:
    return np.asarray(spsolve(A.tocsc(), b), dtype=float)


class GaussSeidel:
    """Reusable lexicographic forward Gauss-Seidel smoother."""

    def __init__(self, A: sparse.spmatrix):
        self.lower = sparse.tril(A, format="csr").tocsc()
        self.upper = sparse.triu(A, 1, format="csr")
        if np.any(self.lower.diagonal() == 0):
            raise ValueError("Zero diagonal makes Gauss-Seidel undefined")

    def sweep(self, x: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.asarray(spsolve_triangular(self.lower,
                                             b - self.upper @ x,
                                             lower=True), dtype=float)

    def smooth(self, x: np.ndarray, b: np.ndarray,
               sweeps: int = 1) -> np.ndarray:
        for _ in range(sweeps):
            x = self.sweep(x, b)
        return x


def solve_gauss_seidel(A: sparse.spmatrix, b: np.ndarray, *,
                       tol: float = 1e-8, max_cycles: int = 2000,
                       x0: np.ndarray | None = None) -> SolverResult:
    _validate_iterations(tol, max_cycles)
    x = np.zeros_like(b) if x0 is None else np.asarray(x0, float).copy()
    smooth = GaussSeidel(A)
    history = [relative_residual(A, b, x)]
    while history[-1] > tol and len(history) - 1 < max_cycles:
        x = smooth.sweep(x, b)
        history.append(relative_residual(A, b, x))
        if not np.isfinite(history[-1]) or history[-1] > 1e8 * max(1.0, history[0]):
            break
    return SolverResult(x, np.asarray(history), history[-1] <= tol)


def _validate_iterations(tol: float, max_cycles: int) -> None:
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("Tolerance must be finite and positive")
    if not isinstance(max_cycles, int) or max_cycles < 0:
        raise ValueError("max_cycles must be a nonnegative integer")


@dataclass
class _Level:
    A: sparse.csr_matrix
    smoother: GaussSeidel | None
    P: sparse.csr_matrix | None
    R: sparse.csr_matrix | None
    lu: object | None


class Multigrid:
    """Galerkin hierarchy for N divisible by 2**(levels-1).

    Matrices are rediscretised only on the finest level; coarser operators
    are A_coarse=R@A_fine@P. On the coarsest level an LU solve is used.
    """

    def __init__(self, A: sparse.spmatrix, N: int, *, levels: int = 3,
                 pre_sweeps: int = 1, post_sweeps: int = 1):
        if isinstance(levels, bool) or not isinstance(levels, int) or levels < 2:
            raise ValueError("levels must be an integer >= 2")
        if N < 2**(levels-1) or N % 2**(levels-1):
            raise ValueError("N must be divisible by 2**(levels-1)")
        if A.shape != ((N-1)**2, (N-1)**2):
            raise ValueError("A has wrong dimensions for this grid")
        if pre_sweeps < 0 or post_sweeps < 0:
            raise ValueError("Smoothing sweeps must be nonnegative")
        self.pre_sweeps = pre_sweeps
        self.post_sweeps = post_sweeps
        self.levels: list[_Level] = []
        current_A = A.tocsr()
        current_N = N
        for i in range(levels):
            if i == levels - 1:
                self.levels.append(_Level(current_A, None, None, None,
                                          splu(current_A.tocsc())))
            else:
                P, R = transfer_2d(current_N)
                self.levels.append(_Level(current_A, GaussSeidel(current_A), P, R, None))
                current_A = (R @ current_A @ P).tocsr()
                current_N //= 2

    @property
    def A(self) -> sparse.csr_matrix:
        return self.levels[0].A

    def cycle(self, b: np.ndarray, x: np.ndarray | None = None,
              *, kind: str = "V") -> np.ndarray:
        if kind not in ("V", "W"):
            raise ValueError("kind must be 'V' or 'W'")
        gamma = 1 if kind == "V" else 2
        current = np.zeros_like(b) if x is None else np.asarray(x, float).copy()
        return self._cycle(0, np.asarray(b, float), current, gamma)

    def _cycle(self, depth: int, b: np.ndarray, x: np.ndarray,
               gamma: int) -> np.ndarray:
        lev = self.levels[depth]
        if lev.lu is not None:
            return np.asarray(lev.lu.solve(b), float)
        assert lev.smoother is not None and lev.R is not None and lev.P is not None
        x = lev.smoother.smooth(x, b, self.pre_sweeps)
        coarse_rhs = lev.R @ (b - lev.A @ x)
        coarse_e = np.zeros_like(coarse_rhs)
        for _ in range(gamma):
            coarse_e = self._cycle(depth + 1, coarse_rhs, coarse_e, gamma)
        x = x + lev.P @ coarse_e
        return lev.smoother.smooth(x, b, self.post_sweeps)

    def solve(self, b: np.ndarray, *, kind: str = "V", tol: float = 1e-8,
              max_cycles: int = 100, x0: np.ndarray | None = None) -> SolverResult:
        _validate_iterations(tol, max_cycles)
        x = np.zeros_like(b) if x0 is None else np.asarray(x0, float).copy()
        history = [relative_residual(self.A, b, x)]
        while history[-1] > tol and len(history) - 1 < max_cycles:
            x = self.cycle(b, x, kind=kind)
            history.append(relative_residual(self.A, b, x))
            if not np.isfinite(history[-1]):
                break
        return SolverResult(x, np.asarray(history), history[-1] <= tol)
