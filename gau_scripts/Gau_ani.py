#!/usr/bin/env python3
"""ANI-2x ONIOM External 接口；直接调用 bin/calculators.py（不使用 server）"""
import sys
sys.path.insert(0, '/share/home/CodeQ/Benchmark')
import torch
import torchani
from ase.io import read
from bin.calculators import METHODS
from bin.constants import EV_HA, BOHR_TO_ANGSTROM
from bin.gaussian_external import get_external_coord, write_xyz, write_external_output

METHOD = 'ani_2x'

if __name__ == '__main__':
    filein, fileout = sys.argv[2], sys.argv[3]
    ele, coordlist, atom_charges, deriva, charge, spin = get_external_coord(filein)
    xyz_file = write_xyz(ele, coordlist, charge, spin)  # 唯一文件名，防并发覆盖
    atoms = read(xyz_file)

    ene_ev = METHODS[METHOD](atoms, charge, spin)

    grad = None
    if deriva == 1:
        model = torchani.models.ANI2x(periodic_table_index=True).to(
            torch.device('cpu')).double()
        species = torch.from_numpy(atoms.get_atomic_numbers()).unsqueeze(0)
        coords = torch.from_numpy(
            atoms.get_positions().astype(np.float64)
        ).requires_grad_(True).unsqueeze(0)
        model((species, coords)).energies.backward()
        # ANI 坐标为 Å，梯度需换算成 Hartree/Bohr（与旧脚本一致）
        grad = coords.grad.numpy() * BOHR_TO_ANGSTROM

    write_external_output(fileout, ene_ev / EV_HA, grad, len(ele), deriva)
