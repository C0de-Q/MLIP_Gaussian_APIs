"""
calculators/_orb.py — ORB-Mol v3 conservative omol / v2
"""

from constants import server_device
from ._registry import register_method, register_gradient, get_cached_model


@register_method('orbmol')
def compute_orbmol(atoms, charge, spin, base=None):
    """Orb v3 conservative omol model."""
    import torch
    from orb_models.forcefield import atomic_system, pretrained

    device = torch.device(server_device())
    orbff = get_cached_model(
        ('orbmol', str(device)),
        lambda: pretrained.orb_v3_conservative_omol(
            device=device, precision='float32-high'),
    )
    atoms.info["charge"] = charge
    atoms.info["spin"] = spin
    graph = atomic_system.ase_atoms_to_atom_graphs(
        atoms, orbff.system_config, device=device,
    )
    result = orbff.predict(graph, split=False)
    return float(result["energy"])  # eV


@register_gradient('orbmol')
def grad_orbmol(atoms, charge, spin, base=None):
    """Orb v3 conservative omol gradient."""
    import torch
    from orb_models.forcefield import atomic_system, pretrained

    device = torch.device(server_device())
    orbff = get_cached_model(
        ('orbmol', str(device)),
        lambda: pretrained.orb_v3_conservative_omol(
            device=device, precision='float32-high'),
    )
    atoms.info['charge'] = charge
    atoms.info['spin'] = spin
    graph = atomic_system.ase_atoms_to_atom_graphs(
        atoms, orbff.system_config, device=device)
    result = orbff.predict(graph, split=False)
    forces = result['grad_forces'].detach().cpu().numpy()
    return [[-f[0], -f[1], -f[2]] for f in forces]


@register_method('orbmol_v2')
def compute_orbmol_v2(atoms, charge, spin, base=None, device=None):
    """Orb v2 model (orbmol_v2 factory plus its atoms adapter).

    device may be 'cuda:N' or 'cpu'; by default it comes from MLIP_SERVER_DEVICE.
    """
    import torch
    from orb_models.forcefield.pretrained import orbmol_v2

    if device is None:
        device = server_device()
    model, atoms_adapter = get_cached_model(
        ('orbmol_v2', device),
        lambda: orbmol_v2(device=torch.device(device)),
    )
    atoms.info["charge"] = int(charge)
    atoms.info["spin"] = int(spin)
    graph = atoms_adapter.from_ase_atoms(atoms).to(device)
    result = model.predict(graph, split=False, compute_forces=True)
    energy = float(result["energy"].cpu().detach())
    if "forces" in result:
        atoms.arrays["forces"] = result["forces"].cpu().detach().numpy()
    return energy  # eV


@register_gradient('orbmol_v2')
def grad_orbmol_v2(atoms, charge, spin, base=None):
    """Orb v2 gradient, reusing the forces that compute_orbmol_v2 stored."""
    forces = atoms.arrays.get('forces')
    if forces is None:
        raise RuntimeError('compute_orbmol_v2 did not store forces, so no gradient is available')
    return [[-f[0], -f[1], -f[2]] for f in forces]
