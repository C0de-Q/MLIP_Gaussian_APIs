#!/usr/bin/env python3
"""ANI-2x-D4 (MLatom CLI) ONIOM External 接口；直接调用 bin/calculators.py（不使用 server）"""
import sys
sys.path.insert(0, '/share/home/CodeQ/Benchmark')
import numpy as np
from ase.io import read
from bin.calculators import METHODS
from bin.constants import EV_HA, BOHR_TO_ANGSTROM
from bin.gaussian_external import (get_external_coord, unique_scratch_base,
                                   write_xyz, write_external_output)

METHOD = 'd4ani'


def _read_mlatom_grad(natoms, path):
    """读取 MLatom 梯度文件 → Hartree/Bohr 梯度（复刻旧 wrapper）。"""
    grad = np.zeros((natoms, 3), dtype=np.float64)
    with open(path, 'r') as fr:
        fr.readline()
        fr.readline()
        for i in range(natoms):
            parts = fr.readline().split()
            grad[i][0] = float(parts[0]) * BOHR_TO_ANGSTROM
            grad[i][1] = float(parts[1]) * BOHR_TO_ANGSTROM
            grad[i][2] = float(parts[2]) * BOHR_TO_ANGSTROM
    return grad


if __name__ == '__main__':
    filein, fileout = sys.argv[2], sys.argv[3]
    ele, coordlist, atom_charges, deriva, charge, spin = get_external_coord(filein)
    base = unique_scratch_base()
    xyz_file = write_xyz(ele, coordlist, charge, spin, f'{base}.xyz')
    atoms = read(xyz_file)

    # METHODS['d4ani'] 运行 MLatom CLI（唯一前缀的 yestfile/ygradxyzestfile）
    ene_ev = METHODS[METHOD](atoms, charge, spin, base=base)

    grad = _read_mlatom_grad(len(ele), f'{base}_grad.dat') if deriva == 1 else None
    write_external_output(fileout, ene_ev / EV_HA, grad, len(ele), deriva)
