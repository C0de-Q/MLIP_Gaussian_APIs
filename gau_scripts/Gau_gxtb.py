#!/usr/bin/env python3
"""GFN2-xTB (subprocess --gxtb) ONIOM External 接口；直接调用 bin/calculators.py（不使用 server）"""
import sys
sys.path.insert(0, '/share/home/CodeQ/Benchmark')
from ase.io import read
from bin.calculators import METHODS
from bin.constants import EV_HA
from bin.gaussian_external import (get_external_coord, unique_scratch_base,
                                   write_xyz, write_external_output)

METHOD = 'gxtb'

if __name__ == '__main__':
    filein, fileout = sys.argv[2], sys.argv[3]
    ele, coordlist, atom_charges, deriva, charge, spin = get_external_coord(filein)
    base = unique_scratch_base()
    xyz_file = write_xyz(ele, coordlist, charge, spin, f'{base}.xyz')
    atoms = read(xyz_file)

    # METHODS['gxtb'] 在独立临时目录运行 xtb --gxtb，梯度存为 <base>_gradient
    ene_ev = METHODS[METHOD](atoms, charge, spin, base=base)

    grad = None
    if deriva == 1:
        grad = []
        with open(f'{base}_gradient', 'r') as f:
            for line in f.readlines()[2:]:
                stripped = line.strip()
                if stripped.startswith('$end'):
                    break
                if not stripped:
                    continue
                parts = stripped.split()
                if len(parts) == 3:
                    grad.append([float(parts[0]), float(parts[1]),
                                 float(parts[2])])

    write_external_output(fileout, ene_ev / EV_HA, grad, len(ele), deriva)
