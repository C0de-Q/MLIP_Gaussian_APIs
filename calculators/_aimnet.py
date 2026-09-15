"""
calculators/_aimnet.py — AIMNet2 (charge/mult at construction, so a custom function)
"""

from ._registry import register_method, get_cached_model


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
    return atoms.get_potential_energy()  # eV
