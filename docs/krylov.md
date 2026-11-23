# Krylov-accelerated multigrid for the indefinite case

This is an **additional** experiment for the existing 2D manufactured problem. The original discretization, right-hand side, Gauss–Seidel, Galerkin V/W-cycles, figures, and benchmark claims are left unchanged.

## Motivation and mathematical formulation

For Dirichlet conditions on the unit square, the discretized operator is

\[
A_h(c)=-\Delta_h+cI.
\]

At \(c=-20\), the continuous smallest Dirichlet Laplacian eigenvalue is \(2\pi^2\approx19.739\), and the resulting operator is indefinite and close to a resonance. A classical three-level Galerkin V-cycle may fail when a coarse-grid correction poorly approximates low-frequency modes. In the existing experiment, the \(N=16\) and \(N=32\) three-grid V-cycle configurations fail.

We introduce an **auxiliary real-shifted positive-definite operator**

\[
P_h=-\Delta_h+c_{\mathrm{aux}}I
    =A_h(c)+(c_{\mathrm{aux}}-c)I,
\qquad c_{\mathrm{aux}}\geq 0.
\]

The default is \(c_{\mathrm{aux}}=20\), which adds \(40I\) for \(c=-20\). One **zero-initialized** V-cycle applied to \(P_h\) approximates \(P_h^{-1}\) as a *fixed linear left preconditioner* for restarted GMRES applied to the **original** indefinite equation \(A_h(c)x=b\). The input matrix and its right-hand side are not shifted when measuring accuracy. Although the auxiliary matrix is positive definite, its Galerkin coarse matrices and smoother are only an approximate inverse; no spectral-equivalence or grid-independent iteration bound for the indefinite operator is claimed.

This is not complex-shifted Laplacian preconditioning, an indefinite multigrid cure, or a general-purpose Helmholtz solver. The preconditioner is deliberately simple to illustrate Krylov acceleration and the significance of the operator being preconditioned.

## Reproduce

From the repository root (after `python -m pip install -e '.[all]'`):

```bash
python -m pytest tests/test_krylov.py -v
python experiments/gmres_indefinite.py --grids 16 32 64 --coefficient -20 --surrogate-coefficient 20
```

The script writes `results/krylov_comparison.csv` (a generated result) and prints convergence status, **actual Arnoldi iteration count**, and **true unpreconditioned relative residual** for:

1. Unshifted standalone three-grid V-cycle (at most 25 cycles).
2. Restarted GMRES without preconditioning (restart 30, at most 20 restart cycles).
3. Restarted GMRES with one V-cycle of the shifted operator as preconditioner (same outer budget).

The acceptance criterion is \(\lVert b-Ax\rVert_2/\lVert b\rVert_2\leq10^{-8}\) with SciPy reporting convergence. For zero right-hand sides, the residual helper follows the existing package's denominator convention. SciPy's GMRES callback exposes **preconditioned** relative norms; the code stores them separately and uses a separately recomputed *true* relative residual to determine success.

### Reference run for the new experiment

With Python 3.13, SciPy 1.17.0 and NumPy 2.3.5, the following was observed:

| N | Unshifted V-cycle | GMRES, no preconditioner | GMRES, shifted V-cycle preconditioner |
|---:|---|---|---|
| 16 | Failed after 25 cycles | Converged in 105 Arnoldi steps | Converged in 17 Arnoldi steps |
| 32 | Failed after 25 cycles | Not converged after 600 steps | Converged in 17 Arnoldi steps |
| 64 | Converged in 11 cycles | Not converged after 600 steps | Converged in 17 Arnoldi steps |

These values result from the stated **fixed tolerance and iteration budgets**, not a universal comparison. The 11 V-cycles at \(N=64\) cannot be compared directly with 17 GMRES steps: one multigrid cycle and one Krylov iteration incur different work, and a preconditioned GMRES step additionally applies multigrid. Single-run timings in the output CSV are exploratory and must not be interpreted as reliable speedups. Convergence can differ with SciPy versions or different parameters.

## Verification and limitations

`tests/test_krylov.py` checks the numerical solutions against a sparse direct solve, including the known negative-reaction failure case; verifies the surrogate shift and zero-started fixed preconditioning; checks V/W-cycle linearity; tests zero right-hand sides, exactly solved initial guesses, reproducibility and nonconvergence reporting; and rejects invalid parameters and dimensions.

This experiment does **not** perform performance tuning, Krylov restart optimization, mixed-precision analysis, eigenvalue deflation, complex shifts, alternate smoothers, or large-scale PDE benchmarks. Near resonance, small residuals can coexist with larger solution errors; the independent sparse direct comparison helps detect that issue in the tests.
