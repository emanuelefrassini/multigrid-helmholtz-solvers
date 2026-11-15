"""Bilinear interpolation and scaled-transpose full-weighting restriction.

Transfer operators act on INTERIOR ERROR variables with homogeneous Dirichlet
conditions. The physical nonzero boundary conditions enter the fine-grid RHS.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse


def prolongation_1d(N: int) -> sparse.csr_matrix:
    """Linear interpolation coarse (N/2-1) -> fine (N-1) interior nodes."""
    if not isinstance(N, int) or N < 4 or N % 2:
        raise ValueError("N must be an even integer >= 4")
    ncoarse = N // 2 - 1
    rows, cols, values = [], [], []
    for i in range(1, N):  # 1-based physical grid coordinate i/N
        if i % 2 == 0:
            j = i // 2  # 1-based coarse coordinate
            rows.append(i - 1); cols.append(j - 1); values.append(1.0)
        else:
            for j in (i // 2, i // 2 + 1):
                if 1 <= j <= ncoarse:
                    rows.append(i - 1); cols.append(j - 1); values.append(0.5)
    return sparse.csr_matrix((values, (rows, cols)),
                             shape=(N - 1, ncoarse))


def transfer_2d(N: int) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Return P (bilinear prolongation), R=(1/4)P.T (full weighting)."""
    P1 = prolongation_1d(N)
    P = sparse.kron(P1, P1, format="csr")
    R = (0.25 * P.T).tocsr()
    return P, R
