#!/usr/bin/env python3
"""
gaussian_external.py — Gaussian External 接口共用工具

供 gau_scripts/Gau_*.py 和 calculators.py 共用。
处理 Gaussian External 格式的输入解析、XYZ 文件写入和输出文件写入。

Gaussian External 调用格式:
    Gau_xxx.py  layer  inputfile  outputfile  [msgfile  [fchkfile]]

inputfile 格式:
    第 1 行: atoms deriva charge spin
    后续行:  atomic_number  x(Bohr)  y(Bohr)  z(Bohr)  [point_charge]
"""

import os
import sys
import numpy as np
from .constants import BOHR_TO_ANGSTROM, ELEMENTS


def get_external_coord(filein):
    """解析 Gaussian External 接口写入的输入文件。

    格式:
        第 1 行: atoms deriva charge spin
        后续行:  atomic_number  x(Bohr)  y(Bohr)  z(Bohr)  [point_charge]

    第 5 列 point_charge 为可选，不存在时填 0.0。

    Returns:
        ele:            np.ndarray (atoms,) int — 原子序数
        coordlist:      np.ndarray (atoms, 3) float — 坐标 (Å)
        atom_charges:   np.ndarray (atoms,) float — 点电荷
        deriva:         int — 0=能量, 1=能量+梯度
        charge:         int — 总电荷
        spin:           int — 自旋多重度
    """
    with open(filein, 'r') as f:
        atoms, deriva, charge, spin = [int(s) for s in f.readline().split()]
        ele = np.zeros(atoms, dtype=int)
        coordlist = np.zeros((atoms, 3), dtype=float)
        atom_charges = np.zeros(atoms, dtype=float)

        for i in range(atoms):
            parts = f.readline().split()
            ele[i] = int(parts[0])
            coordlist[i][0] = float(parts[1]) * BOHR_TO_ANGSTROM
            coordlist[i][1] = float(parts[2]) * BOHR_TO_ANGSTROM
            coordlist[i][2] = float(parts[3]) * BOHR_TO_ANGSTROM
            atom_charges[i] = float(parts[4]) if len(parts) >= 5 else 0.0

    return ele, coordlist, atom_charges, deriva, charge, spin


def unique_scratch_base(prefix='gau'):
    """生成当前进程唯一的临时文件前缀（PBS 作业号 + PID）。

    并发作业在同一目录运行时，若共用固定名 gau_mlatom.xyz 会被互相
    覆盖导致解析失败，因此所有临时文件都改用该前缀。
    """
    job = os.environ.get('PBS_JOBID', '').split('.')[0]
    tag = job if job else 'local'
    return f'{prefix}_{tag}_{os.getpid()}'


def write_xyz(ele, coordlist, charge, spin, xyz_path=None):
    """将原子坐标写入 .xyz 文件（供 ASE Calculator 读入）。

    输出格式:
        <n_atoms>
        <charge> <spin>
        <element> <x> <y> <z>
        ...

    xyz_path 缺省时自动生成唯一文件名（gau_mlatom_<作业号>_<pid>.xyz），
    避免并发作业共用固定名 gau_mlatom.xyz 互相覆盖。
    """
    if xyz_path is None:
        xyz_path = unique_scratch_base() + '.xyz'
    n_atoms = len(ele)
    lines = [str(n_atoms), f'{charge} {spin}']

    for i in range(n_atoms):
        symbol = ELEMENTS[int(ele[i])]
        x, y, z = coordlist[i]
        lines.append(f'{symbol:2s} {x:20.12f} {y:20.12f} {z:20.12f}')

    with open(xyz_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    return xyz_path


def write_external_output(fileout, energy, grad, natoms, deriva):
    """写入 Gaussian External 期望的输出格式。

    Args:
        fileout: 输出文件路径
        energy:  能量 (Hartree)
        grad:    梯度列表 (Hartree/Bohr)，deriva==0 时可传 None
        natoms:  原子数
        deriva:  0=仅能量, 1=能量+梯度
    """
    # 防御：MLIP 返回的 numpy 数组能量统一转成 Python 标量再格式化
    energy = float(np.asarray(energy).reshape(-1)[0])
    with open(fileout, 'w') as fw:
        # 第 1 行: energy, dipole_x, dipole_y, dipole_z
        fw.write(f'{energy:20.12E}{0.0:20.12E}{0.0:20.12E}{0.0:20.12E}\n')

        if deriva == 1 and grad is not None and len(grad) == natoms:
            for g in grad:
                fw.write(f'{g[0]:20.12E}{g[1]:20.12E}{g[2]:20.12E}\n')
        else:
            for _ in range(natoms):
                fw.write(f'{0.0:20.12E}{0.0:20.12E}{0.0:20.12E}\n')
