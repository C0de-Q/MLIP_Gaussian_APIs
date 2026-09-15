#!/usr/bin/env python3
"""Debug tool (not an MLIP interface): turn a Gaussian External input file into
an .xyz file carrying charge and spin, and write zero energy and gradients back
to Gaussian.

Usage:
    Gau_debug.py  layer  inputfile  outputfile   # as called by Gaussian
    Gau_debug.py  inputfile                      # only write the xyz file
"""
import os
import sys
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from gaussian_external import parse_gaussian_args
from gaussian_external import get_external_coord
from constants import ELEMENTS


def write_xyz_with_charges(ele, coordlist, atom_charges, charge, spin, xyz_path):
    n_atoms = len(ele)
    lines = [str(n_atoms), f'{charge} {spin}']
    for i in range(n_atoms):
        symbol = ELEMENTS[int(ele[i])]
        x, y, z = coordlist[i]
        q = atom_charges[i]
        lines.append(f'{symbol:2s} {x:20.12f} {y:20.12f} {z:20.12f} {q:12.6f}')
    with open(xyz_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return xyz_path


if __name__ == '__main__':
    n_args = len(sys.argv) - 1
    if n_args == 1:
        filein, fileout = sys.argv[1], None
    elif n_args >= 3:
        filein, fileout, _, _ = parse_gaussian_args()
    else:
        print(__doc__)
        sys.exit(1)

    ele, coordlist, atom_charges, deriva, charge, spin = get_external_coord(filein)

    base = os.path.splitext(os.path.basename(filein))[0]
    xyz_file = base + '_test.xyz'
    write_xyz_with_charges(ele, coordlist, atom_charges, charge, spin, xyz_file)

    print(f'Charge: {charge}, Spin: {spin}')
    print(f'Atoms: {len(ele)}')
    print(f'Sum of atom charges: {atom_charges.sum():.6f}')
    print(f'XYZ file saved to: {xyz_file}')

    if fileout:
        with open(fileout, 'w') as fw:
            fw.write(f'{0.0:20.12E}{0.0:20.12E}{0.0:20.12E}{0.0:20.12E}\n')
            for _ in range(len(ele)):
                fw.write(f'{0.0:20.12E}{0.0:20.12E}{0.0:20.12E}\n')
