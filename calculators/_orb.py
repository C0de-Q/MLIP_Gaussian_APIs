"""
calculators/_orb.py — ORB-Moland ORB-Mol v2

Upstream usage: the pretrained factories return (model, atoms_adapter), the ASE
calculator takes both, and the OrbMol models require the total charge and spin
multiplicity in atoms.info. Energy and forces then come from one evaluation.
"""

import os

from constants import server_device
from ._registry import register_method, register_gradient, get_cached_model


@register_method('orbmol')
def compute_orbmol(atoms, charge, spin, base=None):
    """ORB-Mol v3 conservative omol."""
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.inference.calculator import ORBCalculator

    device = server_device()
    atoms.info['charge'] = int(charge)
    atoms.info['spin'] = int(spin)

    def build():
        options = dict(device=device, precision='float32-high')
        weights = os.environ.get('MLIP_MODEL_ORBMOL')
        if weights:
            options['weights_path'] = weights
        orbff, atoms_adapter = pretrained.orb_v3_conservative_omol(**options)
        return ORBCalculator(orbff, atoms_adapter=atoms_adapter, device=device)

    key = ('orbmol', str(device))
    atoms.calc = get_cached_model(key, build)
    return float(atoms.get_potential_energy())          # eV


@register_gradient('orbmol')
def grad_orbmol(atoms, charge, spin, base=None):
    """dE/dr in eV/Å, from the evaluation the energy call already did."""
    if atoms.calc is None:
        compute_orbmol(atoms, charge, spin, base)
    return [[-f[0], -f[1], -f[2]] for f in atoms.get_forces()]   # eV/Å


@register_method('orbmol_v2')
def compute_orbmol_v2(atoms, charge, spin, base=None):
    """ORB-Mol v2, the architecture with long-range electrostatics."""
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.inference.calculator import ORBCalculator

    device = server_device()
    atoms.info['charge'] = int(charge)
    atoms.info['spin'] = int(spin)

    def build():
        options = dict(device=device, precision='float32-high')
        weights = os.environ.get('MLIP_MODEL_ORBMOL_V2')
        if weights:
            options['weights_path'] = weights
        orbff, atoms_adapter = pretrained.orbmol_v2(**options)
        return ORBCalculator(orbff, atoms_adapter=atoms_adapter, device=device)

    key = ('orbmol_v2', str(device))
    atoms.calc = get_cached_model(key, build)
    return float(atoms.get_potential_energy())          # eV


@register_gradient('orbmol_v2')
def grad_orbmol_v2(atoms, charge, spin, base=None):
    """dE/dr in eV/Å, from the evaluation the energy call already did."""
    if atoms.calc is None:
        compute_orbmol_v2(atoms, charge, spin, base)
    return [[-f[0], -f[1], -f[2]] for f in atoms.get_forces()]   # eV/Å
