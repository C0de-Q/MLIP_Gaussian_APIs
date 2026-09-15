"""
calculators/_ani.py — the ANI family through torchani (ANI-1x / 1ccx / 2x)
"""

import numpy as np
from constants import EV_HA, server_device
from ._registry import register_method, register_gradient, get_cached_model


_ANI_MODELS = {
    '1x':   'ANI1x',
    '1ccx': 'ANI1ccx',
    '2x':   'ANI2x',
}


def _ani_model(ani_variant):
    import torchani

    if ani_variant not in _ANI_MODELS:
        raise ValueError(
            f"Unknown ANI variant: {ani_variant}. Choose from: {list(_ANI_MODELS)}")
    device = server_device()
    # Weights are loaded once per device and reused by every later request.
    return get_cached_model(
        ('ani', ani_variant, device),
        lambda: getattr(torchani.models, _ANI_MODELS[ani_variant])(
            periodic_table_index=True).to(device).double(),
    )


def _compute_ani(atoms, ani_variant):
    """ANI energy, converted from Hartree to eV."""
    import torch

    device = server_device()
    model = _ani_model(ani_variant)
    species = torch.from_numpy(atoms.get_atomic_numbers()).to(device).unsqueeze(0)
    coordinates = torch.from_numpy(
        atoms.get_positions().astype(np.float64)
    ).to(device).requires_grad_(True).unsqueeze(0)

    energy = model((species, coordinates)).energies
    return energy.item() * EV_HA  # Hartree → eV


def _grad_ani(atoms, ani_variant):
    """ANI gradient: torchani energies are in Hartree and coordinates in Å, so
    dE/dcoord in Ha/Å is multiplied by EV_HA to get eV/Å."""
    import torch

    device = server_device()
    model = _ani_model(ani_variant)
    species = torch.from_numpy(atoms.get_atomic_numbers()).to(device).unsqueeze(0)
    # unsqueeze comes before requires_grad_ on purpose: otherwise the batched
    # tensor is not a leaf and backward() would leave coords_batch.grad empty.
    coords_batch = torch.from_numpy(
        atoms.get_positions().astype(np.float64)
    ).to(device).unsqueeze(0).requires_grad_(True)
    model((species, coords_batch)).energies.backward()
    grad = coords_batch.grad
    if grad is None:
        raise RuntimeError('torchani did not produce a gradient for these coordinates')
    # Drop the batch dimension torchani works with: callers expect (natoms, 3).
    return (grad[0].cpu().numpy() * EV_HA).tolist()


@register_method('ani_1x')
def compute_ani_1x(atoms, charge, spin, base=None):
    """ANI-1x (H C N O, wB97X/6-31G(d))。"""
    return _compute_ani(atoms, '1x')


@register_gradient('ani_1x')
def grad_ani_1x(atoms, charge, spin, base=None):
    return _grad_ani(atoms, '1x')


@register_method('ani_1ccx')
def compute_ani_1ccx(atoms, charge, spin, base=None):
    """ANI-1ccx (H C N O, CCSD(T)*/CBS)。"""
    return _compute_ani(atoms, '1ccx')


@register_gradient('ani_1ccx')
def grad_ani_1ccx(atoms, charge, spin, base=None):
    return _grad_ani(atoms, '1ccx')


@register_method('ani_2x')
def compute_ani_2x(atoms, charge, spin, base=None):
    """ANI-2x (H C N O F S Cl, wB97X/6-31G(d))。"""
    return _compute_ani(atoms, '2x')


@register_gradient('ani_2x')
def grad_ani_2x(atoms, charge, spin, base=None):
    return _grad_ani(atoms, '2x')
