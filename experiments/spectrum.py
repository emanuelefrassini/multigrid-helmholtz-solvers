"""Inspect how the smallest algebraic eigenvalue changes across the hierarchy.

Useful for understanding multigrid difficulties near resonance when c=-20.
Small dense coarse matrices are used only for diagnostics; production multigrid
uses sparse matrices and a sparse coarsest-level LU factorisation.
"""
import argparse
from pathlib import Path
import csv

from scipy.sparse.linalg import eigsh
from multigrid_helmholtz import Multigrid, discretize


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--grids',type=int,nargs='+',default=[16,32,64])
    parser.add_argument('--coefficients',type=float,nargs='+',default=[-20])
    parser.add_argument('--levels',type=int,default=3)
    parser.add_argument('--out',type=Path,default=Path('results/spectrum.csv'))
    a=parser.parse_args()
    rows=[]
    for c in a.coefficients:
        for N in a.grids:
            p=discretize(N,c)
            hier=Multigrid(p.A,N,levels=a.levels)
            for level,lev in enumerate(hier.levels):
                mineig=float(eigsh(lev.A,k=1,which='SA',return_eigenvectors=False)[0])
                rows.append({'N':N,'c':c,'level':level,
                             'matrix_size':lev.A.shape[0],
                             'smallest_eigenvalue':mineig})
                print(f'N={N:3d} c={c:g} depth={level} '
                      f'size={lev.A.shape[0]:5d} min_eig={mineig:+.7f}')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    print('Saved',a.out)


if __name__=='__main__':
    main()
