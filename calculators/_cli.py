"""
calculators/_cli.py — models computed through an external CLI (MLatom / aitomic)

Covers d4ani (MLatom) and aiqm3 (aitomic). Each call runs the external program in
a temporary directory and keeps the result in memory, so a run leaves no .xyz,
.inp, .out, or .dat files behind in the working directory.
"""

import os
import shutil
import tempfile
import threading

import numpy as np

from constants import CLI_PATHS, EV_HA
from gaussian_external import unique_scratch_base
from ._registry import register_method, register_gradient

# Energy and gradient of the most recent CLI runs, keyed by scratch base. The
# gradient is produced by the same run as the energy and the server requests it
# in a separate call with that base, so caching it also keeps that call file-free.
_RESULTS = {}
_RESULTS_LOCK = threading.Lock()
_MAX_RESULTS = 8


def _store_result(base, energy_ev, grad):
    with _RESULTS_LOCK:
        _RESULTS[base] = (energy_ev, grad)
        while len(_RESULTS) > _MAX_RESULTS:
            _RESULTS.pop(next(iter(_RESULTS)))


def _cached_gradient(name, base):
    with _RESULTS_LOCK:
        result = _RESULTS.get(base)
    if result is None:
        raise RuntimeError(
            f'{name} reads its gradient from the energy evaluation that produced '
            f'base={base}, and no such evaluation is cached')
    if result[1] is None:
        raise RuntimeError(f'{name} produced no gradient for base={base}')
    return result[1]


def _run_cli(program, method_name, atoms, charge, spin, base=None,
             extra_input=()):
    """Run an external CLI for one structure and return its energy in eV.

    Every input and output file lives in a temporary directory that is removed
    before this function returns. The gradient from the same run is cached for
    the matching grad_<method> call.
    """
    from ase.io import write as ase_write

    if base is None:
        base = unique_scratch_base(program)
    tmpdir = tempfile.mkdtemp(prefix=os.path.basename(base) + '_')
    try:
        xyz_path = os.path.join(tmpdir, 'structure.xyz')
        inp_path = os.path.join(tmpdir, 'input.inp')
        out_path = os.path.join(tmpdir, 'stdout.txt')
        ene_path = os.path.join(tmpdir, 'energy.dat')
        grad_path = os.path.join(tmpdir, 'gradient.dat')

        ase_write(xyz_path, atoms)
        with open(inp_path, 'w') as fw:
            fw.write(method_name + '\n')
            for line in extra_input:
                fw.write(line + '\n')
            fw.write(f'xyzfile={xyz_path}\n')
            fw.write(f'yestfile={ene_path}\n')
            fw.write(f'ygradxyzestfile={grad_path}\n')

        os.system(f'{CLI_PATHS[program]} {inp_path} > {out_path}')

        with open(ene_path, 'r') as fr:
            energy_ev = float(fr.readline().strip()) * EV_HA  # Hartree to eV

        try:
            grad = _read_cli_gradient(len(atoms), grad_path)
        except OSError:
            grad = None      # an energy-only run may produce no gradient file
        _store_result(base, energy_ev, grad)
        return energy_ev
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _read_cli_gradient(natoms, path):
    """Read an MLatom-style gradient file (Hartree/Å) and return dE/dr in eV/Å."""
    grad = np.zeros((natoms, 3), dtype=np.float64)
    with open(path, 'r') as fr:
        fr.readline()
        fr.readline()
        for i in range(natoms):
            parts = fr.readline().split()
            grad[i][0] = float(parts[0]) * EV_HA
            grad[i][1] = float(parts[1]) * EV_HA
            grad[i][2] = float(parts[2]) * EV_HA
    return grad


@register_method('d4ani')
def compute_d4ani(atoms, charge, spin, base=None):
    """ANI-2x-D4 through the MLatom CLI.

    MLatom is an external program, so every step starts a new process that loads
    the model again and it cannot stay resident in the server. Switch to an
    equivalent Python package if that becomes the bottleneck.
    """
    return _run_cli('mlatom', 'ANI-2x-D4', atoms, charge, spin, base)


@register_gradient('d4ani')
def grad_d4ani(atoms, charge, spin, base=None):
    return _cached_gradient('d4ani', base)


@register_method('aiqm3')
def compute_aiqm3(atoms, charge, spin, base=None):
    """AIQM3 through the aitomic CLI."""
    return _run_cli('aitomic', 'AIQM3', atoms, charge, spin, base,
                    extra_input=(f'charges={charge}',
                                 f'multiplicities={spin}'))


@register_gradient('aiqm3')
def grad_aiqm3(atoms, charge, spin, base=None):
    return _cached_gradient('aiqm3', base)
