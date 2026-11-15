"""Manufactured 2-D reaction-diffusion (Helmholtz-type) boundary-value problem.

    -Delta(u) + c*u = f,   u|boundary = exact_u.

The same manufactured exact solution appears in the supplied MATLAB coursework.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def exact_u(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """u(x,y) = x(1-x)y^3(1-y) + exp(x)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return x * (1 - x) * y**3 * (1 - y) + np.exp(x)


def forcing(x: ArrayLike, y: ArrayLike, c: float) -> np.ndarray:
    """f=-Delta(u)+c*u, differentiated analytically from exact_u."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    p = x * (1 - x)
    q = y**3 * (1 - y)
    minus_laplacian = 2 * q - p * (6 * y - 12 * y**2) - np.exp(x)
    return minus_laplacian + c * (p * q + np.exp(x))
