#!/usr/bin/env python3
"""AIMNet2 + D4 色散校正 ONIOM External 接口；直接调用 bin/calculators.py（不使用 server）"""
import sys
import os
import json
import shutil
import subprocess
import tempfile
sys.path.insert(0, '/share/home/CodeQ/Benchmark')
import numpy as np
from ase.io import read
from bin.calculators import METHODS
from bin.constants import EV_HA, FORCE_UNIT_CONST
from bin.gaussian_external import get_external_coord, write_xyz, write_external_output

METHOD = 'aimnet2_d4'

if __name__ == '__main__':
    filein, fileout = sys.argv[2], sys.argv[3]
    ele, coordlist, atom_charges, deriva, charge, spin = get_external_coord(filein)
    xyz_file = write_xyz(ele, coordlist, charge, spin)  # 唯一文件名
    atoms = read(xyz_file)

    # AIMNet2 部分（calculators.py 的 METHODS['aimnet2']）
    ene_ev = METHODS['aimnet2'](atoms, charge, spin)
    forces = atoms.get_forces() if deriva == 1 else None

    # D4 色散校正：在独立临时目录运行 dftd4bin，避免 dftd4.json 并发冲突
    tmpdir = tempfile.mkdtemp(prefix='dftd4_', dir='.')
    try:
        shutil.copy2(xyz_file, os.path.join(tmpdir, os.path.basename(xyz_file)))
        args = [os.environ['dftd4bin'], os.path.basename(xyz_file),
                '-f', 'wb97m', '-c', str(charge), '-s', '-s',
                '--noedisp', '--json']
        if deriva == 1:
            args.append('--grad')
        subprocess.Popen(args, cwd=tmpdir, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, universal_newlines=True).wait()
        with open(os.path.join(tmpdir, 'dftd4.json'), 'r') as f:
            d4 = json.load(f)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    energy = ene_ev / EV_HA + float(d4['energy'])

    grad = None
    if deriva == 1:
        d4_grad = np.array(d4['gradient']).reshape(-1, 3)
        grad = []
        for i in range(len(ele)):
            grad.append([
                -forces[i][0] / FORCE_UNIT_CONST + d4_grad[i][0] / 0.5291772109,
                -forces[i][1] / FORCE_UNIT_CONST + d4_grad[i][1] / 0.5291772109,
                -forces[i][2] / FORCE_UNIT_CONST + d4_grad[i][2] / 0.5291772109,
            ])

    write_external_output(fileout, energy, grad, len(ele), deriva)
