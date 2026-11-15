# Numerical formulation and verification

## Origin of the model

The uploaded scientific-computing coursework contains MATLAB routines named
`helmotzsolution.m`, `helmotzgridc_20.m`, `gaussseidel.m`,
`twogridv_c20.m`, `twogridv_c_20.m`, `threegridvcycle.m`, and
`threegridWcycle.m`. They investigate a manufactured 2-D elliptic boundary
problem using Gauss–Seidel, two-grid, and three-grid cycles. The original report
examines `c=20` and `c=-20` and notes difficulties with negative `c`.

This Python project retains that mathematical **problem and comparison** but
uses its own interior-only sparse assembly and a consistent Galerkin correction
hierarchy. It is **not** a literal MATLAB translation, and its measured iteration
counts need not reproduce the historical plots.

## Manufactured exact solution and forcing

On \(\Omega=(0,1)^2\), solve

\[
 -\Delta u + cu = f \quad \text{in }\Omega,
 \qquad u|_{\partial\Omega}=g.
\]

Set

\[
 u_\mathrm{exact}(x,y)=p(x)q(y)+e^x,
 \quad p(x)=x(1-x),\quad q(y)=y^3(1-y),
 \quad g=u_\mathrm{exact}|_{\partial\Omega}.
\]

We differentiate the manufactured solution analytically, obtaining

\[
 f(x,y)=2q(y)-p(x)(6y-12y^2)-e^x+c[p(x)q(y)+e^x].
\]

This formula works for **both** `c=20` and `c=-20`; it avoids manually
maintaining separate, error-prone RHS expressions. Finite-difference checks
of the analytic forcing are part of the test suite.

## Five-point finite differences

For \(h=1/N\), interior nodes have indices \(i,j=1,\ldots,N-1\).
We use the five-point stencil

\[
 (A_h u)_{ij}=\frac{4u_{ij}-u_{i-1,j}-u_{i+1,j}-u_{i,j-1}-u_{i,j+1}}{h^2}
 + c u_{ij}.
\]

Only the \((N-1)^2\) **interior** degrees of freedom are stored as unknowns.
Known Dirichlet neighbour values are moved to the right-hand side as
\(g/h^2\). We use row-major array ordering with the `x` index varying fastest.
The matrix is a symmetric sparse five-point operator, constructed with
Kronecker sums; no dense iteration matrix is formed.

The discrete direct reference uses `scipy.sparse.linalg.spsolve`. The error
\(\|u_h-u_\mathrm{exact}\|_\infty\) measures **discretisation error**;
relative residual \(\|b-Ax\|_2/\|b\|_2\) measures **algebraic convergence**.
These are distinct quantities.

## Gauss–Seidel and multigrid

Lexicographic forward Gauss–Seidel solves
\((D+L)x^{k+1}=b-Ux^k\), where `L` and `U` are the strict lower and upper
triangular parts of \(A\). It is implemented with sparse triangular solves,
not by explicitly assembling \((D+L)^{-1}\).

For multigrid, the fine-level operator is the manufactured five-point matrix.
At each level, prolongation \(P\) uses **bilinear interpolation** of the
homogeneous-Dirichlet error; restriction is \(R=\tfrac14 P^T\), equivalent
to full weighting on the interior. The next coarser operator is Galerkin:

\[
 A_{2h}=RA_hP.
\]

A cycle applies one forward GS pre-sweep, restricts the residual, approximately
solves the coarse **error equation**, prolongs the correction, then applies
one GS post-sweep. The two-grid solver has two levels. Three-grid V-cycles
visit the next coarse level once; three-grid W-cycles make two recursive
visits. The coarsest level uses sparse LU.

All solvers start from zero, use relative residual tolerance `1e-8` by
default and report whether it is reached before the cycle limit. For severe
divergence the iteration may be terminated at a relative residual exceeding
`1e8` as a numerical safety cutoff.

## Why the negative reaction coefficient matters

For \(c=20\), \(A_h\) is symmetric positive definite. For \(c=-20\), the
continuous Dirichlet Laplacian has its smallest eigenvalue
\(2\pi^2\approx19.739\), so \(-\Delta-20 I\) is **indefinite** and close
to a resonance. Classical Gauss–Seidel and multigrid are no longer assured
to converge. In particular, more work per cycle (W instead of V) does not
guarantee a better outcome.

`experiments/spectrum.py` calculates the smallest eigenvalue on each of the
Galerkin levels. On this implementation, the `N=16, c=-20` three-level
hierarchy has a smallest eigenvalue that becomes **positive** on the coarsest
level, unlike on the fine grid. For `N=32`, the coarsest operator has an
eigenvalue very close to zero. Both effects help explain the breakdown of
naive coarse-grid correction. Other hierarchies can be stable, and this
observation should not be generalised to all indefinite multigrid schemes.

## Test strategy and interpretation

Tests verify analytic forcing, symmetric sparse matrix assembly, Dirichlet
boundary handling, approximate second-order grid convergence, transfer
weights, Galerkin symmetry, positive-case solver convergence and agreement
with the independent sparse direct solution. Selected negative-case
configurations check both convergence and a known divergent case.

Convergence plots use **sweeps** for standalone GS and **cycles** for
multigrid. They are not equal units of computational work; iteration-count
comparisons do **not** establish wall-time speedups. `experiments/run_all.py`
logs single-run elapsed times for exploratory context only, without
controlled repetition, hardware disclosure or performance claims.
