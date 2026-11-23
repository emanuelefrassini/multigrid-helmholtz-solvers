"""GMRES and fixed shifted-multigrid preconditioning for Helmholtz-type systems.

For A=-Delta_h+cI, construct the auxiliary SPD operator
P=-Delta_h+c_aux I = A+(c_aux-c)I with c_aux >= 0. One
V/W-cycle on P approximates the left preconditioner P^{-1}.

This is a real-shifted elliptic surrogate, not a generally robust
Helmholtz preconditioner. Convergence must be checked on the original A.
"""
from __future__ import annotations

from dataclasses import dataclass
from inspect import signature

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, gmres

from .solvers import Multigrid, relative_residual


class ShiftedMultigridPreconditioner:
    """A *fixed linear* multigrid preconditioner for a shifted surrogate.

    ``reaction`` is the coefficient in the original ``-Delta_h + c*I``;
    ``surrogate_reaction`` is the coefficient in the auxiliary operator.
    Nonnegative surrogate reaction guarantees a positive-definite auxiliary
    finite-difference matrix for the supported homogeneous error grid.
    """

    def __init__(
        self,
        A: sparse.spmatrix,
        N: int,
        reaction: float,
        *,
        surrogate_reaction: float = 20.0,
        levels: int = 3,
        cycle: str = "V",
    ) -> None:
        if isinstance(N, bool) or not isinstance(N, (int, np.integer)) or N < 4:
            raise ValueError("N must be an integer >= 4")
        size = (int(N) - 1) ** 2
        if A.shape != (size, size):
            raise ValueError("A has the wrong dimensions for N")
        if not np.isfinite(reaction) or not np.isfinite(surrogate_reaction):
            raise ValueError("reaction coefficients must be finite")
        if surrogate_reaction < 0:
            raise ValueError("surrogate_reaction must be nonnegative")
        if cycle not in ("V", "W"):
            raise ValueError("cycle must be 'V' or 'W'")
        self.surrogate_matrix = (
            A.tocsr() + (surrogate_reaction - reaction) * sparse.eye(size, format="csr")
        ).tocsr()
        self.multigrid = Multigrid(self.surrogate_matrix, int(N), levels=levels)
        self.cycle_kind = cycle
        self.operator = LinearOperator(
            (size, size),
            matvec=self.apply,
            dtype=np.float64,
        )

    def apply(self, residual: np.ndarray) -> np.ndarray:
        """Apply one *zero-started* cycle: required for a fixed GMRES operator."""
        v = np.asarray(residual, dtype=float)
        if v.ndim != 1 or v.shape[0] != self.surrogate_matrix.shape[0]:
            raise ValueError("preconditioner input has wrong shape")
        return np.asarray(self.multigrid.cycle(v, kind=self.cycle_kind), dtype=float)


@dataclass(frozen=True)
class GMRESResult:
    solution: np.ndarray
    converged: bool
    info: int
    iterations: int  # inner Arnoldi steps, NOT restart cycles
    true_relative_residual: float
    preconditioned_residual_history: np.ndarray


def gmres_solve(
    A: sparse.spmatrix,
    b: np.ndarray,
    *,
    preconditioner: ShiftedMultigridPreconditioner | None = None,
    tol: float = 1e-8,
    restart: int = 30,
    maxiter: int = 20,
    x0: np.ndarray | None = None,
) -> GMRESResult:
    """Solve Ax=b with restarted left-preconditioned GMRES.

    ``maxiter`` counts outer restart cycles (SciPy ``callback_type='pr_norm'``).
    ``iterations`` counts the actual inner Arnoldi steps. Callback values
    are *preconditioned* residual norms and must not be compared directly to
    the true unpreconditioned relative residual used for acceptance.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    right = np.asarray(b, dtype=float)
    if right.shape != (A.shape[0],) or not np.all(np.isfinite(right)):
        raise ValueError("b must be a finite vector matching A")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")
    if isinstance(restart, bool) or not isinstance(restart, (int, np.integer)) or restart < 1:
        raise ValueError("restart must be a positive integer")
    if isinstance(maxiter, bool) or not isinstance(maxiter, (int, np.integer)) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer")
    initial = None if x0 is None else np.asarray(x0, dtype=float)
    if initial is not None and (initial.shape != right.shape or not np.all(np.isfinite(initial))):
        raise ValueError("x0 must be finite and match b")
    if preconditioner is not None and preconditioner.operator.shape != A.shape:
        raise ValueError("preconditioner shape mismatch")

    history: list[float] = []
    kwargs = dict(
        x0=initial, atol=0.0, restart=int(restart), maxiter=int(maxiter),
        M=None if preconditioner is None else preconditioner.operator,
        callback=lambda r: history.append(float(r)), callback_type="pr_norm",
    )
    # SciPy < 1.12 calls this parameter 'tol'; newer releases use 'rtol'.
    kwargs["rtol" if "rtol" in signature(gmres).parameters else "tol"] = float(tol)
    x, info = gmres(A, right, **kwargs)
    x = np.asarray(x, dtype=float)
    residual = relative_residual(A, right, x)
    if not np.isfinite(residual):
        raise ArithmeticError("GMRES produced a non-finite true relative residual")
    return GMRESResult(
        solution=x,
        converged=(int(info) == 0 and residual <= tol * (1.0 + 1e-7)),
        info=int(info),
        iterations=len(history),
        true_relative_residual=residual,
        preconditioned_residual_history=np.asarray(history),
    )
