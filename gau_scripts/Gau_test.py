#!/usr/bin/env python3
"""
Gau_test.py — 从 Gaussian External 输入生成带 charge/spin 的 .xyz 文件

Gaussian 调用方式:
    Gau_test.py  layer  inputfile  outputfile
    Gau_test.py  inputfile  outputfile
    Gau_test.py  inputfile

生成的 .xyz 文件格式:
    <n_atoms>
    <charge> <spin>
    <element> <x> <y> <z> <atom_charge>
    ...

该 xyz 文件可直接用于 MLIP_xyz.py 做单点能计算。
"""
import sys
import os
import numpy as np
sys.path.insert(0, '/share/home/CodeQ/Benchmark')
from bin.gaussian_external import get_external_coord
from bin.constants import ELEMENTS


def write_xyz_with_charges(ele, coordlist, atom_charges, charge, spin, xyz_path):
    """写入带点电荷列的 .xyz 文件。"""
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
    if len(sys.argv) >= 4:
        filein = sys.argv[2]
        fileout = sys.argv[3]
    elif len(sys.argv) == 3:
        filein = sys.argv[1]
        fileout = sys.argv[2]
    elif len(sys.argv) == 2:
        filein = sys.argv[1]
        fileout = None
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
