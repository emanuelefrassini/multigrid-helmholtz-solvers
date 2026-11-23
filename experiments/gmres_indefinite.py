"""Compare classical multigrid and (shifted-multigrid) GMRES.

Usage: python experiments/gmres_indefinite.py --grids 16 32 64
One-shot elapsed times are diagnostic only; do not interpret as benchmarks.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import perf_counter

import numpy as np

from multigrid_helmholtz import Multigrid, discretize, direct_solve, relative_residual
from multigrid_helmholtz.krylov import ShiftedMultigridPreconditioner, gmres_solve


def run_problem(N: int, c: float, *, surrogate_c: float = 20.0,
                tol: float = 1e-8, levels: int = 3, restart: int = 30,
                maxiter: int = 20, mg_cycles: int = 25) -> list[dict]:
    p = discretize(N, c)
    exact_discrete = direct_solve(p.A, p.rhs)
    reference_norm = max(np.linalg.norm(exact_discrete), 1.0)
    rows = []

    def record(method: str, solution, iterations: int, converged: bool,
               elapsed: float, *, info: int | None = None):
        residual = relative_residual(p.A, p.rhs, solution)
        rows.append(dict(
            N=N, c=c, method=method, converged=bool(converged),
            iterations=int(iterations), true_relative_residual=residual,
            relative_solution_error_vs_direct=float(np.linalg.norm(solution-exact_discrete)/reference_norm),
            elapsed_ms=1000.0*elapsed, info=info,
        ))

    start = perf_counter()
    mg = Multigrid(p.A, N, levels=levels)
    mg_result = mg.solve(p.rhs, tol=tol, max_cycles=mg_cycles)
    record("unshifted_multigrid_V", mg_result.solution, mg_result.iterations,
           mg_result.converged, perf_counter()-start)

    start = perf_counter()
    raw_gmres = gmres_solve(p.A, p.rhs, tol=tol, restart=restart, maxiter=maxiter)
    record("gmres_no_preconditioner", raw_gmres.solution, raw_gmres.iterations,
           raw_gmres.converged, perf_counter()-start, info=raw_gmres.info)

    start = perf_counter()
    shifted = ShiftedMultigridPreconditioner(
        p.A, N, reaction=c, surrogate_reaction=surrogate_c, levels=levels)
    accelerated = gmres_solve(p.A, p.rhs, preconditioner=shifted,
                              tol=tol, restart=restart, maxiter=maxiter)
    record("gmres_shifted_multigrid_V", accelerated.solution, accelerated.iterations,
           accelerated.converged, perf_counter()-start, info=accelerated.info)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--coefficient", type=float, default=-20.0)
    ap.add_argument("--surrogate-coefficient", type=float, default=20.0)
    ap.add_argument("--levels", type=int, default=3)
    ap.add_argument("--restart", type=int, default=30)
    ap.add_argument("--maxiter", type=int, default=20)
    ap.add_argument("--mg-cycles", type=int, default=25)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--output", type=Path, default=Path("results/krylov_comparison.csv"))
    args = ap.parse_args()
    rows = []
    for N in args.grids:
        rows.extend(run_problem(
            N, args.coefficient, surrogate_c=args.surrogate_coefficient,
            levels=args.levels, restart=args.restart, maxiter=args.maxiter,
            mg_cycles=args.mg_cycles, tol=args.tol))
    for row in rows:
        print(f"N={row['N']:3d} {row['method']:27s} "
              f"iter={row['iterations']:4d} converged={str(row['converged']):5s} "
              f"true_rel_res={row['true_relative_residual']:.2e}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
