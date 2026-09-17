"""
calculators/_ani.py — the ANI family through torchani (ANI-1x / 1ccx / 2x)
"""

import numpy as np
from constants import EV_HA, server_device
from ._registry import (register_method, register_gradient, get_cached_model,
                        store_result, take_result)


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


def _ani_inputs(atoms, device):
    """Species and coordinate tensors for one forward pass.

    unsqueeze comes before requires_grad_ on purpose: otherwise the batched
    coordinate tensor is not a leaf and autograd would produce no gradient for it.
    """
    import torch

    species = torch.from_numpy(atoms.get_atomic_numbers()).to(device).unsqueeze(0)
    coords = torch.from_numpy(
        atoms.get_positions().astype(np.float64)
    ).to(device).unsqueeze(0).requires_grad_(True)
    return species, coords


def _compute_ani(atoms, ani_variant, base=None):
    """ANI energy in eV; the forward pass is handed to the gradient call.

    Sending both requests from one evaluation is worth it — a backward costs about
    as much as a forward here. The tensors stay in the registry's small cache until
    grad_<method> takes them, or until an entry is evicted.
    """
    device = server_device()
    model = _ani_model(ani_variant)
    species, coords = _ani_inputs(atoms, device)
    energies = model((species, coords)).energies
    store_result(base, energies=energies, coords=coords)
    return energies.item() * EV_HA  # Hartree → eV


def _grad_ani(atoms, ani_variant, base=None):
    """ANI gradient: torchani energies are in Hartree and coordinates in Å, so
    dE/dcoord in Ha/Å is multiplied by EV_HA to get eV/Å.

    Reuses the forward pass stored by the energy call; without one it runs its own.
    """
    import torch

    cached = take_result(base)
    if cached is not None:
        energies, coords = cached['energies'], cached['coords']
    else:
        device = server_device()
        species, coords = _ani_inputs(atoms, device)
        energies = _ani_model(ani_variant)((species, coords)).energies
    grad = torch.autograd.grad(energies.sum(), coords)[0]
    # Drop the batch dimension torchani works with: callers expect (natoms, 3).
    return (grad[0].cpu().numpy() * EV_HA).tolist()


@register_method('ani_1x')
def compute_ani_1x(atoms, charge, spin, base=None):
    """ANI-1x (H C N O, wB97X/6-31G(d))。"""
    return _compute_ani(atoms, '1x', base)


@register_gradient('ani_1x')
def grad_ani_1x(atoms, charge, spin, base=None):
    return _grad_ani(atoms, '1x', base)


@register_method('ani_1ccx')
def compute_ani_1ccx(atoms, charge, spin, base=None):
    """ANI-1ccx (H C N O, CCSD(T)*/CBS)。"""
    return _compute_ani(atoms, '1ccx', base)


@register_gradient('ani_1ccx')
def grad_ani_1ccx(atoms, charge, spin, base=None):
    return _grad_ani(atoms, '1ccx', base)


@register_method('ani_2x')
def compute_ani_2x(atoms, charge, spin, base=None):
    """ANI-2x (H C N O F S Cl, wB97X/6-31G(d))。"""
    return _compute_ani(atoms, '2x', base)


@register_gradient('ani_2x')
def grad_ani_2x(atoms, charge, spin, base=None):
    return _grad_ani(atoms, '2x', base)
