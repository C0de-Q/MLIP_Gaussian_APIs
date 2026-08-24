#!/usr/bin/env python3
"""ORB-Mol ONIOM External 接口；直接调用 bin/calculators.py（不使用 server）"""
import sys
sys.path.insert(0, '/share/home/CodeQ/Benchmark')
import torch
from ase.io import read
from bin.calculators import METHODS
from bin.constants import EV_HA, FORCE_UNIT_CONST
from bin.gaussian_external import get_external_coord, write_xyz, write_external_output

METHOD = 'orbmol'

if __name__ == '__main__':
    filein, fileout = sys.argv[2], sys.argv[3]
    ele, coordlist, atom_charges, deriva, charge, spin = get_external_coord(filein)
    xyz_file = write_xyz(ele, coordlist, charge, spin)  # 唯一文件名，防并发覆盖
    atoms = read(xyz_file)

    ene_ev = METHODS[METHOD](atoms, charge, spin)

    grad = None
    if deriva == 1:
        from orb_models.forcefield import atomic_system, pretrained
        orbff = pretrained.orb_v3_conservative_omol(
            device=torch.device('cpu'), precision='float32-high')
        atoms.info['charge'] = charge
        atoms.info['spin'] = spin
        graph = atomic_system.ase_atoms_to_atom_graphs(
            atoms, orbff.system_config, device=torch.device('cpu'))
        result = orbff.predict(graph, split=False)
        forces = result['grad_forces'].detach().cpu().numpy()
        grad = [[-f[0] / FORCE_UNIT_CONST,
                 -f[1] / FORCE_UNIT_CONST,
                 -f[2] / FORCE_UNIT_CONST] for f in forces]

    write_external_output(fileout, ene_ev / EV_HA, grad, len(ele), deriva)
