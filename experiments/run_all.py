"""Reproducible residual histories and manufactured-solution errors.

Example:
    python experiments/run_all.py --grids 16 32 64 --coefficients 20 -20

The printed times are ONE exploratory wall-clock run per configuration, not
controlled benchmarks. Re-run on target hardware for timing comparisons.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from multigrid_helmholtz import (Multigrid, discretize, direct_solve,
                                 relative_residual, solve_gauss_seidel)

ROOT = Path(__file__).resolve().parents[1]


def run_one(N: int, c: float, *, tol: float, gs_max: int,
            mg_max: int):
    prob = discretize(N,c)
    t0=perf_counter()
    exact_discrete = direct_solve(prob.A,prob.rhs)
    direct_time=1000*(perf_counter()-t0)
    rows=[dict(N=N,c=c,method='direct',iterations=1,converged=True,
               relative_residual=relative_residual(prob.A,prob.rhs,exact_discrete),
               max_error_vs_exact=float(np.max(np.abs(exact_discrete-prob.exact_interior))),
               elapsed_ms=direct_time)]
    histories={}
    methods=[('gauss_seidel',lambda: solve_gauss_seidel(prob.A,prob.rhs,tol=tol,max_cycles=gs_max))]
    for levels,name,kind in ((2,'two_grid_V','V'),(3,'three_grid_V','V'),(3,'three_grid_W','W')):
        if N % (2**(levels-1)) == 0:
            mg=Multigrid(prob.A,N,levels=levels)
            methods.append((name,lambda mg=mg,kind=kind: mg.solve(prob.rhs,kind=kind,tol=tol,max_cycles=mg_max)))
    for name,solve in methods:
        start=perf_counter()
        result=solve()
        elapsed=1000*(perf_counter()-start)
        rows.append(dict(N=N,c=c,method=name,iterations=result.iterations,
                         converged=result.converged,
                         relative_residual=float(result.residuals[-1]),
                         max_error_vs_exact=float(np.max(np.abs(result.solution-prob.exact_interior))),
                         elapsed_ms=elapsed))
        histories[name]=result.residuals
    return rows,histories


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--grids',type=int,nargs='+',default=[16,32,64])
    parser.add_argument('--coefficients',type=float,nargs='+',default=[20.0,-20.0])
    parser.add_argument('--tol',type=float,default=1e-8)
    parser.add_argument('--gs-max',type=int,default=1000)
    parser.add_argument('--mg-max',type=int,default=50)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results')
    parser.add_argument('--figure-dir',type=Path,default=ROOT/'figures')
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    args.figure_dir.mkdir(parents=True,exist_ok=True)
    rows=[]
    histories={}
    for c in args.coefficients:
        for N in args.grids:
            r,h=run_one(N,c,tol=args.tol,gs_max=args.gs_max,mg_max=args.mg_max)
            rows+=r
            histories[N,c]=h
            for line in r:
                print(f"N={N:3d} c={c:5g} {line['method']:15s} "
                      f"iter={line['iterations']:4d} converged={str(line['converged']):5s} "
                      f"relative_res={line['relative_residual']:.3e} "
                      f"max_error={line['max_error_vs_exact']:.3e}")
    with (args.output_dir/'solver_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    colors={'gauss_seidel':'#777777','two_grid_V':'#1D4ED8',
            'three_grid_V':'#0F766E','three_grid_W':'#C05621'}
    for (N,c),runs in histories.items():
        fig,ax=plt.subplots(figsize=(8.2,5))
        for method,history in runs.items():
            ax.semilogy(np.arange(len(history)),np.minimum(np.maximum(history,1e-16),1e8),
                        label=method.replace('_',' '),lw=2,color=colors[method])
        ax.axhline(args.tol,linestyle='--',color='black',alpha=.5,label='target tolerance')
        ax.set_xscale('symlog',linthresh=15)
        ax.set_xlim(left=0)
        ax.set_xlabel('GS sweeps or multigrid cycles (unequal work per step)')
        if c < 0:
            ax.set_ylim(1e-10, 2e8)
            ax.text(.02,.03,'Divergent traces clipped at 10⁸',transform=ax.transAxes,
                    color='#555555',fontsize=9)
        ax.set_ylabel('Relative residual $||b-Ax||_2/||b||_2$')
        ax.set_title(f'Helmholtz-type system: N={N}, c={c:g}')
        ax.grid(alpha=.25,which='both');ax.legend(fontsize=9)
        fig.tight_layout()
        label=f'neg{abs(c):g}' if c<0 else f'pos{c:g}'
        filename=args.figure_dir/f'residual_N{N}_c{label}.png'
        fig.savefig(filename,dpi=165);plt.close(fig)
    # Grid refinement for the sparse direct reference, independent of iterative convergence.
    fig,ax=plt.subplots(figsize=(7.5,4.5))
    for c in args.coefficients:
        sample=[r for r in rows if r['c']==c and r['method']=='direct']
        sample.sort(key=lambda r:r['N'])
        ax.loglog([1/r['N'] for r in sample],
                  [r['max_error_vs_exact'] for r in sample], 'o-',lw=2,
                  label=f'c={c:g}')
    ax.set_xlabel('Grid spacing h=1/N')
    ax.set_ylabel('Max absolute error vs manufactured solution')
    ax.set_title('Grid refinement: five-point finite differences')
    ax.grid(alpha=.2,which='both');ax.legend();fig.tight_layout()
    fig.savefig(args.figure_dir/'grid_refinement.png',dpi=165);plt.close(fig)
    print('Saved',args.output_dir/'solver_results.csv')
    print('Saved figures in',args.figure_dir)


if __name__=='__main__':
    main()
