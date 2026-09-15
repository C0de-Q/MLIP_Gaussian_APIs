#!/usr/bin/env python3
"""
gaussian_external.py — helpers shared by the Gaussian External scripts

Used by gau_scripts/Gau_*.py and the calculators package. It parses Gaussian
External input and writes the External output file; nothing here creates files
of its own.

Gaussian calls
    Gau_xxx.py  layer  inputfile  outputfile  [msgfile  [fchkfile]]

and the input file contains
    line 1:   atoms deriva charge spin
    then one line per atom:
              atomic_number  x(Bohr)  y(Bohr)  z(Bohr)  [point_charge]
"""

import os
import sys
import time
import numpy as np
from constants import BOHR_TO_ANGSTROM, EV_HA, FORCE_UNIT_CONST


def parse_gaussian_args(argv=None):
    """Parse Gaussian External command line arguments.

    The standard call passes the layer first, which is how Gaussian invokes
    Gau_*.py:
        Gau_xxx.py  layer  inputfile  outputfile  [msgfile  [fchkfile]]

    For manual debugging the layer may be omitted:
        Gau_xxx.py  inputfile  outputfile

    Returns:
        (filein, fileout, msgfile, fchkfile)
    """
    if argv is None:
        argv = sys.argv
    n = len(argv) - 1  # number of arguments after the script name

    if n == 2:
        filein, fileout = argv[1], argv[2]
    elif n >= 3:
        filein, fileout = argv[2], argv[3]
    else:
        raise SystemExit(
            'Usage: Gau_xxx.py [layer] inputfile outputfile [msgfile [fchkfile]]\n'
            f'Got {n} argument(s): {argv[1:]}'
        )

    msgfile = argv[4] if n >= 4 else None
    fchkfile = argv[5] if n >= 5 else None
    return filein, fileout, msgfile, fchkfile


def get_external_coord(filein):
    """Parse the input file written by the Gaussian External interface.

    Layout:
        line 1:   atoms deriva charge spin
        per atom: atomic_number  x(Bohr)  y(Bohr)  z(Bohr)  [point_charge]

    The fifth column (point charge) is optional and defaults to 0.0.

    Returns:
        ele:            np.ndarray (atoms,) int — atomic numbers
        coordlist:      np.ndarray (atoms, 3) float — coordinates in Å
        atom_charges:   np.ndarray (atoms,) float — point charges
        deriva:         int — 0 = energy only, 1 = energy and gradient
        charge:         int — total charge
        spin:           int — spin multiplicity
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


_scratch_seq = 0


def unique_scratch_base(prefix='gau'):
    """Return a scratch-file prefix unique to this process and call.

    The prefix combines a nanosecond timestamp (time.time_ns), the process ID,
    and an in-process counter, so it works everywhere (local, PBS, Slurm,
    containers) without relying on PBS_JOBID. A single MLIP evaluation is fast
    and may be called repeatedly in the same process, which is why a counter is
    added on top of the timestamp; concurrent processes are told apart by PID.
    """
    global _scratch_seq
    _scratch_seq += 1
    return f'{prefix}_{time.time_ns()}_{os.getpid()}_{_scratch_seq}'


def write_external_output(fileout, energy, grad, natoms, deriva):
    """Write the output file Gaussian expects from an External call.

    Args:
        fileout: output file path
        energy:  energy in Hartree
        grad:    gradient rows in Hartree/Bohr; may be None when deriva == 0
        natoms:  number of atoms
        deriva:  0 = energy only, 1 = energy and gradient
    """
    # Some models return a numpy array; convert to a Python scalar first.
    energy = float(np.asarray(energy).reshape(-1)[0])
    with open(fileout, 'w') as fw:
        # Line 1: energy, dipole_x, dipole_y, dipole_z
        fw.write(f'{energy:20.12E}{0.0:20.12E}{0.0:20.12E}{0.0:20.12E}\n')

        if deriva == 1 and grad is not None and len(grad) == natoms:
            for g in grad:
                fw.write(f'{g[0]:20.12E}{g[1]:20.12E}{g[2]:20.12E}\n')
        else:
            for _ in range(natoms):
                fw.write(f'{0.0:20.12E}{0.0:20.12E}{0.0:20.12E}\n')


def write_external_output_ev(fileout, energy_ev, forces_ev_ang, deriva, natoms):
    """Write Gaussian External output from values in MLIP units (eV, eV/Å).

    Converts to Hartree and Hartree/Bohr and then reuses
    write_external_output; used by callers such as mlip_server.py that work in
    ASE units.
    """
    energy_ha = float(np.asarray(energy_ev).reshape(-1)[0]) / EV_HA
    if deriva == 1 and forces_ev_ang is not None and len(forces_ev_ang) == natoms:
        grad = [[-f[0] / FORCE_UNIT_CONST,
                 -f[1] / FORCE_UNIT_CONST,
                 -f[2] / FORCE_UNIT_CONST] for f in forces_ev_ang]
    else:
        grad = None
    write_external_output(fileout, energy_ha, grad, natoms, deriva)
