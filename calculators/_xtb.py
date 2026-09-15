"""
calculators/_xtb.py — GFN2-xTB (xtb Python API and the gxtb executable)
"""

from constants import EV_HA, FORCE_UNIT_CONST, require_cli_path
from gaussian_external import unique_scratch_base
from ._registry import register_method, register_gradient


@register_method('xtb')
def compute_xtb(atoms, charge, spin, base=None):
    """GFN2-xTB (xTB Python API)。"""
    # The xtb Calculator carries its coordinates and charge, so it is a per-step
    # object and is not cached. Constructing it is cheap and does not count as
    # reloading a model.
    from xtb.interface import Calculator, Param
    from xtb.libxtb import VERBOSITY_FULL

    numbers = atoms.get_atomic_numbers()
    positions = atoms.get_positions()

    calc = Calculator(
        Param.GFN2xTB, numbers, positions,
        charge=float(charge), uhf=int(spin - 1),
    )
    calc.set_verbosity(VERBOSITY_FULL)
    calc.set_electronic_temperature(300)

    res = calc.singlepoint()
    return res.get_energy() * EV_HA  # Hartree → eV


@register_gradient('xtb')
def grad_xtb(atoms, charge, spin, base=None):
    """GFN2-xTB gradient in eV/Å through the xtb Python API."""
    from xtb.interface import Calculator, Param
    from xtb.libxtb import VERBOSITY_FULL

    calc = Calculator(
        Param.GFN2xTB, atoms.get_atomic_numbers(),
        atoms.get_positions(), charge=float(charge),
        uhf=int(spin - 1),
    )
    calc.set_verbosity(VERBOSITY_FULL)
    # The xtb API returns Hartree/Bohr; convert to eV/Å.
    return calc.singlepoint().get_gradient() * FORCE_UNIT_CONST


@register_method('gxtb')
def compute_gxtb(atoms, charge, spin, base=None):
    """GFN2-xTB through the xtb executable (--gxtb mode)."""
    import os
    import re
    import shutil
    import subprocess
    import tempfile
    from ase.io import write as ase_write

    if base is None:
        base = unique_scratch_base('gxtb')

    # Scratch files go to the system temporary directory, so a run leaves
    # nothing behind in the working directory.
    tmpdir = tempfile.mkdtemp(prefix=os.path.basename(base) + '_')
    try:
        xyz_path = os.path.join(tmpdir, os.path.basename(base) + '.xyz')
        log_path = os.path.join(tmpdir, os.path.basename(base) + '.log')
        ase_write(xyz_path, atoms)

        command = [
            require_cli_path('xtb'), os.path.basename(xyz_path),
            '--gxtb', '--chrg', str(charge), '--norestart', '--grad',
        ]
        with open(log_path, 'w') as log_file:
            subprocess.run(command, cwd=tmpdir, stdout=log_file,
                           stderr=subprocess.PIPE, text=True)

        # Read the energy from the gradient file.
        scf_energy = None
        with open(os.path.join(tmpdir, 'gradient'), 'r') as f:
            lines = f.readlines()
        if len(lines) >= 2:
            match = re.search(r'=[^=]*=\s*(\S+)', lines[1].strip())
            if match:
                scf_energy = float(match.group(1))
        if scf_energy is None:
            raise RuntimeError("Failed to parse gxtb energy from gradient file")
        # Rename gradient to a unique file for the caller to read.
        os.rename(os.path.join(tmpdir, 'gradient'), f'{base}_gradient')
        return scf_energy * EV_HA  # Hartree → eV
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@register_gradient('gxtb')
def grad_gxtb(atoms, charge, spin, base=None):
    """gxtb gradient read from <base>_gradient (Hartree/Bohr converted to eV/Å)."""
    if base is None:
        raise RuntimeError('the gxtb gradient needs the base argument')
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
                grad.append([float(parts[0]) * FORCE_UNIT_CONST,
                             float(parts[1]) * FORCE_UNIT_CONST,
                             float(parts[2]) * FORCE_UNIT_CONST])
    return grad
