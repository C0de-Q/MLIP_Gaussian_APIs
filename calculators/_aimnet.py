"""
calculators/_aimnet.py — AIMNet2 (charge/mult at construction, so a custom function)
"""

import numpy as np

from ._registry import register_method, register_gradient, get_cached_model


@register_method('aimnet2')
def compute_aimnet2(atoms, charge, spin, base=None):
    """AIMNet2 using its built-in 'aimnet2' model name."""
    from aimnet2calc import AIMNet2ASE

    # Cached per (charge, spin): charge and multiplicity stay constant through an
    # optimization, so the model is loaded once.
    atoms.calc = get_cached_model(
        ('aimnet2', int(charge), int(spin)),
        lambda: AIMNet2ASE('aimnet2', charge=charge, mult=spin),
    )
    # aimnet2calc returns array([energy]) instead of a float; the rest of the
    # package expects a scalar (and NumPy 2 refuses to format an array as a float).
    return float(np.asarray(atoms.get_potential_energy()).item())


@register_gradient('aimnet2')
def grad_aimnet2(atoms, charge, spin, base=None):
    """dE/dr in eV/Å, from the calculator the energy call attached to these atoms.

    ASE returns the forces of the evaluation that already happened, so this costs
    no second model run.
    """
    if atoms.calc is None:                  # gradient asked for without an energy call
        compute_aimnet2(atoms, charge, spin, base)
    forces = atoms.get_forces()             # eV/Å
    return [[-f[0], -f[1], -f[2]] for f in forces]
