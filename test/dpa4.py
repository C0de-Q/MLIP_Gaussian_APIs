#!/usr/bin/env python3
"""test/dpa4.py — reference script for the real DPA4 model

A non-toy example of adding an ASE model. It is the same registration as the dpa4
entry in test/custom_models.py and the built-in dpa4 in
calculators/_ase_models.py: a standard ASE calculator (deepmd.calculator.DP) plus
a prepare hook that writes charge_spin into atoms.info.

Run it directly (needs deepmd-kit and DPA4 weights):

    # fill in MLIP_MODEL_DPA4=/data/models/DPA4-xxx.pt in config.env
    python test/dpa4.py

Or run it through the server:

    ./RunMLIPgjf.sh -m dpa4 job.gjf

When a dependency or the weight path is missing, the script reports what to fix
instead of raising a traceback.
"""

import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from constants import require_model_path  # noqa: E402

# Neutral singlet, cation doublet, anion singlet as (charge, spin)
STATES = [(0, 1), (1, 2), (-1, 1)]


def main():
    try:
        from ase.build import molecule
        from deepmd.calculator import DP
    except ImportError as e:
        print(f'Missing dependency: {e}\n  Install deepmd-kit first (see requirements.txt).')
        return 1

    try:
        model = require_model_path('dpa4')
    except RuntimeError as e:
        print(e)
        return 1

    print(f'DPA4 weights: {model}')
    for charge, spin in STATES:
        atoms = molecule('CH2_s1A1d')
        atoms.info.update({'charge_spin': np.array([charge, spin])})
        atoms.calc = DP(model=model)
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        print(f'charge={charge:+d} spin={spin}  '
              f'E={energy:.6f} eV  max|F|={np.abs(forces).max():.6f} eV/Å')
    return 0


if __name__ == '__main__':
    sys.exit(main())
