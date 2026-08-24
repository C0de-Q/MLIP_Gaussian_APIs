#!/usr/bin/env python3
"""GFN2-xTB (Python API) ONIOM External 接口；直接调用 bin/calculators.py（不使用 server）"""
import sys
sys.path.insert(0, '/share/home/CodeQ/Benchmark')
from ase.io import read
from bin.calculators import METHODS
from bin.constants import EV_HA
from bin.gaussian_external import get_external_coord, write_xyz, write_external_output

METHOD = 'xtb'

if __name__ == '__main__':
    filein, fileout = sys.argv[2], sys.argv[3]
    ele, coordlist, atom_charges, deriva, charge, spin = get_external_coord(filein)
    xyz_file = write_xyz(ele, coordlist, charge, spin)  # 唯一文件名，防并发覆盖
    atoms = read(xyz_file)

    ene_ev = METHODS[METHOD](atoms, charge, spin)

    grad = None
    if deriva == 1:
        from xtb.interface import Calculator, Param
        from xtb.libxtb import VERBOSITY_FULL
        calc = Calculator(Param.GFN2xTB, atoms.get_atomic_numbers(),
                          atoms.get_positions(), charge=float(charge),
                          uhf=int(spin - 1))
        calc.set_verbosity(VERBOSITY_FULL)
        grad = calc.singlepoint().get_gradient()  # Hartree/Bohr

    write_external_output(fileout, ene_ev / EV_HA, grad, len(ele), deriva)
