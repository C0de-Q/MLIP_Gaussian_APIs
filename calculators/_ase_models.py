"""
calculators/_ase_models.py — built-in ASE models (MACE and DeepMD DPA families)

Most of these are declarative ASE models (BUILTIN_ASE_MODELS), which the server
loads once through the persistent ASE path. mace_polar is written functionally
because it must translate the molecule to the center-of-mass origin first; it
uses @register_method with get_cached_model and is loaded once as well.
"""

import numpy as np

from constants import require_model_path, server_device
from ._registry import (register_ase_model, register_method,
                        register_gradient, get_cached_model)


BUILTIN_ASE_MODELS = {
    # device=True means "this factory takes a device argument"; its value comes
    # from MLIP_SERVER_DEVICE in config.env (see constants.server_device).
    'mace_omol': dict(module='mace.calculators', factory='mace_omol',
                      model='mace_omol', device=True),
    'mace_off23': dict(module='mace.calculators', factory='mace_off',
                       model='mace_off23', device=True),
    'mace_off24': dict(module='mace.calculators', factory='mace_off',
                       model='mace_off24', device=True),
    'dpa3': dict(module='deepmd.calculator', factory='DP',
                 model='dpa3_omol',
                 prepare=lambda atoms, charge, spin: atoms.info.update(
                     {'fparam': [charge, spin]})),
    'dpa2_drug': dict(module='deepmd.calculator', factory='DP',
                      model='dpa2_drug'),
    # DPA4 is a standard ASE calculator like DPA3; it passes charge and spin
    # through charge_spin rather than fparam.
    'dpa4': dict(module='deepmd.calculator', factory='DP',
                 model='dpa4',
                 prepare=lambda atoms, charge, spin: atoms.info.update(
                     {'charge_spin': np.array([charge, spin])})),
}

for _name, _spec in BUILTIN_ASE_MODELS.items():
    register_ase_model(_name, _spec)


def _translate_to_origin(atoms):
    """Translate the molecule so its center of mass sits at the origin.

    MACE-POLAR requires the origin to be at the center of mass, so this runs
    before both the energy and the gradient evaluation.
    """
    atoms.translate(-atoms.get_center_of_mass())


@register_method('mace_polar')
def compute_mace_polar(atoms, charge, spin, base=None):
    """MACE-POLAR-1-M, written functionally.

    MACE-POLAR requires the origin at the center of mass, so the molecule is
    translated before each evaluation. get_cached_model keeps the model loaded
    once per server process.
    """
    from mace.calculators import mace_polar

    # Translate the molecule to the origin before evaluating.
    _translate_to_origin(atoms)

    atoms.info["charge"] = int(charge)
    atoms.info["spin"] = int(spin)
    atoms.info["external_field"] = [0.0, 0.0, 0.0]

    device = server_device()
    atoms.calc = get_cached_model(
        ('mace_polar', device),
        lambda: mace_polar(model=require_model_path('mace_polar'),
                           device=device, default_dtype='float32'),
    )
    return atoms.get_potential_energy()  # eV


@register_gradient('mace_polar')
def grad_mace_polar(atoms, charge, spin, base=None):
    """MACE-POLAR gradient (same translation to the origin as the energy)."""
    _translate_to_origin(atoms)
    forces = atoms.get_forces()  # eV/Å; the calculator was attached by compute_mace_polar
    return [[-f[0], -f[1], -f[2]] for f in forces]
