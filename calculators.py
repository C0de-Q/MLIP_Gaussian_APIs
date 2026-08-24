#!/usr/bin/env python3
"""
calculators.py — 统一 MLIP 能量计算器集合

所有计算器函数签名统一为:
    def compute_xxx(atoms: ase.Atoms, charge: int, spin: int) -> float:
        ...
        return energy  # eV

方法注册表 METHODS 字典供按名称查找。
"""

import numpy as np
from .constants import EV_HA, FORCE_UNIT_CONST, BOHR_TO_ANGSTROM, MODEL_PATHS, CLI_PATHS


# ═══════════════════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════════════════

def _run_mlatom_simple(method_name, atoms, charge, spin, base=None):
    """MLatom CLI 通用执行器 (simple 格式: <method> / xyzfile= / yestfile=)。

    用于 ANI-2x-D4 等方法。base 为本次调用的唯一临时文件前缀，
    缺省时自动生成，避免并发作业共用固定文件名互相覆盖。
    """
    import os
    from ase.io import write as ase_write

    if base is None:
        from .gaussian_external import unique_scratch_base
        base = unique_scratch_base()
    xyz_path = f'{base}.xyz'
    inp_path = f'{base}.inp'
    out_path = f'{base}.out'
    ene_path = f'{base}_ene.dat'
    grad_path = f'{base}_grad.dat'

    ase_write(xyz_path, atoms)

    with open(inp_path, 'w') as fw:
        fw.write(f'{method_name}\n')
        fw.write(f'xyzfile={xyz_path}\n')
        fw.write(f'yestfile={ene_path}\n')
        fw.write(f'ygradxyzestfile={grad_path}\n')

    os.system(f'{CLI_PATHS["mlatom"]} {inp_path} > {out_path}')

    with open(ene_path, 'r') as fr:
        ene = float(fr.readline().strip())

    return ene * EV_HA  # Hartree → eV


def _run_aitomic(method_name, atoms, charge, spin, base=None):
    """aitomic CLI 执行器 (AIQM3 格式: charges= / multiplicities=)。

    base 为本次调用的唯一临时文件前缀，缺省时自动生成。
    """
    import os
    from ase.io import write as ase_write

    if base is None:
        from .gaussian_external import unique_scratch_base
        base = unique_scratch_base()
    xyz_path = f'{base}.xyz'
    inp_path = f'{base}.inp'
    out_path = f'{base}.out'
    ene_path = f'{base}_ene.dat'
    grad_path = f'{base}_grad.dat'

    ase_write(xyz_path, atoms)

    with open(inp_path, 'w') as fw:
        fw.write(f'{method_name}\n')
        fw.write(f'charges={charge}\n')
        fw.write(f'multiplicities={spin}\n')
        fw.write(f'xyzfile={xyz_path}\n')
        fw.write(f'yestfile={ene_path}\n')
        fw.write(f'ygradxyzestfile={grad_path}\n')

    os.system(f'{CLI_PATHS["aitomic"]} {inp_path} > {out_path}')

    with open(ene_path, 'r') as fr:
        ene = float(fr.readline().strip())

    return ene * EV_HA  # Hartree → eV


def _run_mlatom_useMLmodel(method_name, atoms, charge, spin, base=None):
    """MLatom CLI 执行器 (useMLmodel 格式，用于 MACE OFF24 等)。

    base 为本次调用的唯一临时文件前缀，缺省时自动生成。
    """
    import os
    from ase.io import write as ase_write

    if base is None:
        from .gaussian_external import unique_scratch_base
        base = unique_scratch_base()
    xyz_path = f'{base}.xyz'
    inp_path = f'{base}.inp'
    out_path = f'{base}.out'
    ene_path = f'{base}_ene.dat'

    ase_write(xyz_path, atoms)

    with open(inp_path, 'w') as fw:
        fw.write('useMLmodel\n')
        fw.write(f'MLmodelType=MACE\n')
        fw.write(f'MLmodelIn={MODEL_PATHS.get(method_name, "")}\n')
        fw.write(f'xyzfile={xyz_path}\n')
        fw.write(f'yestfile={ene_path}\n')

    os.system(f'{CLI_PATHS["mlatom"]} {inp_path} > {out_path}')

    with open(ene_path, 'r') as fr:
        ene = float(fr.readline().strip())

    return ene * EV_HA  # Hartree → eV


# ═══════════════════════════════════════════════════════════════════════
#  MACE family
# ═══════════════════════════════════════════════════════════════════════

def compute_mace_polar(atoms, charge, spin):
    """MACE Polar 模型 (极化率)。"""
    from mace.calculators import mace_polar

    atoms.info["charge"] = int(charge)
    atoms.info["spin"] = int(spin)
    atoms.info["external_field"] = [0.0, 0.0, 0.0]
    atoms.calc = mace_polar(
        model=MODEL_PATHS['mace_polar'],
        device='cpu',
        default_dtype="float32",
    )
    return atoms.get_potential_energy()  # eV


def compute_mace_omol(atoms, charge, spin):
    """MACE omol extra-large 模型。"""
    from mace.calculators import mace_omol

    atoms.info["charge"] = int(charge)
    atoms.info["spin"] = int(spin)
    atoms.calc = mace_omol(
        model=MODEL_PATHS['mace_omol'],
        device='cpu',
    )
    return atoms.get_potential_energy()  # eV


def compute_mace_off23(atoms, charge, spin):
    """MACE-OFF23 large 模型 (用于 off23L)。"""
    from mace.calculators import mace_off

    atoms.info["charge"] = charge
    atoms.info["spin"] = spin
    atoms.calc = mace_off(
        model=MODEL_PATHS['mace_off23'],
        device='cpu',
    )
    return atoms.get_potential_energy()  # eV


def compute_mace_off24(atoms, charge, spin):
    """MACE-OFF24 medium 模型 (用于 off24)。"""
    from mace.calculators import mace_off

    atoms.info["charge"] = charge
    atoms.info["spin"] = spin
    atoms.calc = mace_off(
        model=MODEL_PATHS['mace_off24'],
        device='cpu',
    )
    return atoms.get_potential_energy()  # eV


# ═══════════════════════════════════════════════════════════════════════
#  AIMNet2
# ═══════════════════════════════════════════════════════════════════════

def compute_aimnet2(atoms, charge, spin):
    """AIMNet2 模型 (使用内建 'aimnet2' 模型名)。"""
    from aimnet2calc import AIMNet2ASE

    atoms.calc = AIMNet2ASE('aimnet2', charge=charge, mult=spin)
    return atoms.get_potential_energy()  # eV


# ═══════════════════════════════════════════════════════════════════════
#  DeepMD family
# ═══════════════════════════════════════════════════════════════════════

def compute_dpa3(atoms, charge, spin):
    """DPA3 Omol Large (DeepMD)。"""
    from deepmd.calculator import DP as DPCalculator

    atoms.info.update({"fparam": [charge, spin]})
    atoms.calc = DPCalculator(MODEL_PATHS['dpa3_omol'])
    return atoms.get_potential_energy()  # eV


def compute_dpa2_drug(atoms, charge, spin):
    """DPA2 Drug v1 (DeepMD)。"""
    from deepmd.calculator import DP as DPCalculator

    atoms.calc = DPCalculator(MODEL_PATHS['dpa2_drug'])
    return atoms.get_potential_energy()  # eV


# ═══════════════════════════════════════════════════════════════════════
#  Orb
# ═══════════════════════════════════════════════════════════════════════

def compute_orbmol(atoms, charge, spin):
    """Orb v3 conservative omol 模型。"""
    import torch
    from orb_models.forcefield import atomic_system, pretrained

    device = torch.device('cpu')
    orbff = pretrained.orb_v3_conservative_omol(
        device=device,
        precision="float32-high",
    )
    atoms.info["charge"] = charge
    atoms.info["spin"] = spin
    graph = atomic_system.ase_atoms_to_atom_graphs(
        atoms, orbff.system_config, device=device,
    )
    result = orbff.predict(graph, split=False)
    return float(result["energy"])  # eV


def compute_orbmol_v2(atoms, charge, spin, device='cuda:1'):
    """Orb v2 model (orbmol_v2 factory + adapter pattern).

    Uses orbmol_v2() factory returning (model, atoms_adapter),
    then atoms_adapter.from_ase_atoms(atoms).to(device) for graph conversion.
    device 可指定 'cuda:N' 或 'cpu'（默认 cuda:1）。
    """
    import torch
    from orb_models.forcefield.pretrained import orbmol_v2

    device = torch.device(device)
    model, atoms_adapter = orbmol_v2(device=device)
    atoms.info["charge"] = int(charge)
    atoms.info["spin"] = int(spin)
    graph = atoms_adapter.from_ase_atoms(atoms).to(device)
    result = model.predict(graph, split=False, compute_forces=True)
    energy = float(result["energy"].cpu().detach())
    if "forces" in result:
        atoms.arrays["forces"] = result["forces"].cpu().detach().numpy()
    return energy  # eV


# ═══════════════════════════════════════════════════════════════════════
#  xTB family
# ═══════════════════════════════════════════════════════════════════════

def compute_xtb(atoms, charge, spin):
    """GFN2-xTB (xTB Python API)。"""
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


def compute_gxtb(atoms, charge, spin, base=None):
    """GFN2-xTB (subprocess 调用 xtb 可执行文件，--gxtb 模式)。

    在独立临时目录中运行 xtb，并把 gradient 输出改名为唯一的
    <base>_gradient，避免并发作业共用固定文件名。
    """
    import os
    import re
    import shutil
    import subprocess
    import tempfile
    from ase.io import write as ase_write

    if base is None:
        from .gaussian_external import unique_scratch_base
        base = unique_scratch_base()

    tmpdir = tempfile.mkdtemp(prefix=base + '_', dir='.')
    try:
        xyz_path = os.path.join(tmpdir, os.path.basename(base) + '.xyz')
        log_path = os.path.join(tmpdir, os.path.basename(base) + '.log')
        ase_write(xyz_path, atoms)

        command = [
            CLI_PATHS['xtb'], os.path.basename(xyz_path),
            '--gxtb', '--chrg', str(charge), '--norestart', '--grad',
        ]
        with open(log_path, 'w') as log_file:
            subprocess.run(command, cwd=tmpdir, stdout=log_file,
                           stderr=subprocess.PIPE, text=True)

        # 解析 gradient 文件中的能量
        scf_energy = None
        with open(os.path.join(tmpdir, 'gradient'), 'r') as f:
            lines = f.readlines()
        if len(lines) >= 2:
            match = re.search(r'=[^=]*=\s*(\S+)', lines[1].strip())
            if match:
                scf_energy = float(match.group(1))
        if scf_energy is None:
            raise RuntimeError("Failed to parse gxtb energy from gradient file")
        # 把 gradient 改名为唯一文件，供外层脚本读取
        os.rename(os.path.join(tmpdir, 'gradient'), f'{base}_gradient')
        return scf_energy * EV_HA  # Hartree → eV
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════
#  ANI family (torchani)
# ═══════════════════════════════════════════════════════════════════════

def _compute_ani(atoms, ani_variant):
    """ANI 模型内部入口 (torchani 直接调用)。

    Args:
        ani_variant: '1x' | '1ccx' | '2x'

    Returns:
        能量 (eV)
    """
    import torch
    import torchani

    device = torch.device('cpu')
    model_map = {
        '1x':   torchani.models.ANI1x,
        '1ccx': torchani.models.ANI1ccx,
        '2x':   torchani.models.ANI2x,
    }
    if ani_variant not in model_map:
        raise ValueError(f"Unknown ANI variant: {ani_variant}. Choose from: {list(model_map.keys())}")

    model = model_map[ani_variant](periodic_table_index=True).to(device).double()
    species = torch.from_numpy(atoms.get_atomic_numbers()).unsqueeze(0)
    coordinates = torch.from_numpy(
        atoms.get_positions().astype(np.float64)
    ).requires_grad_(True).unsqueeze(0)

    energy = model((species, coordinates)).energies
    return energy.item() * EV_HA  # Hartree → eV


def compute_ani_1x(atoms, charge, spin):
    """ANI-1x (H C N O, wB97X/6-31G(d))。"""
    return _compute_ani(atoms, '1x')


def compute_ani_1ccx(atoms, charge, spin):
    """ANI-1ccx (H C N O, CCSD(T)*/CBS)。"""
    return _compute_ani(atoms, '1ccx')


def compute_ani_2x(atoms, charge, spin):
    """ANI-2x (H C N O F S Cl, wB97X/6-31G(d))。"""
    return _compute_ani(atoms, '2x')


# ═══════════════════════════════════════════════════════════════════════
#  MLatom / aitomic CLI
# ═══════════════════════════════════════════════════════════════════════

def compute_d4ani(atoms, charge, spin, base=None):
    """ANI-2x-D4 (MLatom CLI)。base 为唯一临时文件前缀（可选）。"""
    return _run_mlatom_simple('ANI-2x-D4', atoms, charge, spin, base)


def compute_aiqm3(atoms, charge, spin, base=None):
    """AIQM3 (aitomic CLI)。base 为唯一临时文件前缀（可选）。"""
    return _run_aitomic('AIQM3', atoms, charge, spin, base)


# ═══════════════════════════════════════════════════════════════════════
#  其他模型
# ═══════════════════════════════════════════════════════════════════════

def compute_fennix(atoms, charge, spin):
    """FENNIX 生物分子力场 (fennol)。"""
    from fennol.ase import FENNIXCalculator

    atoms.set_initial_charges(charge)
    atoms.calc = FENNIXCalculator(
        model=MODEL_PATHS['fennix_bio1m'],
        gpu_preprocessing=False,
    )
    return atoms.get_potential_energy()  # eV


def compute_nutmeg(atoms, charge, spin):
    """Nutmeg 大模型。"""
    import torch
    from nutmegpotentials.nutmegcalculator import NutmegCalculator

    device = torch.device('cpu')
    atoms.calc = NutmegCalculator('nutmeg-large', atoms, charge, device)
    return atoms.get_potential_energy()  # eV


def compute_pm6ml(atoms, charge, spin):
    """PM6-ML 半经验修正模型。"""
    from pm6ml import PM6MLCalculator

    atoms.calc = PM6MLCalculator('/share/home/CodeQ/bin/PM6-ML_correction_seed8_best.ckpt')
    return atoms.get_potential_energy()  # eV


def compute_aceff2(atoms, charge, spin):
    """ACE FF v2 力场 (TorchMD-Net)。"""
    from torchmdnet.calculators import TMDNETCalculator

    atoms.info["charge"] = charge
    atoms.calc = TMDNETCalculator(
        model_file=MODEL_PATHS['aceff_v2'],
        device='cpu',
    )
    return atoms.get_potential_energy()  # eV


# ═══════════════════════════════════════════════════════════════════════
#  方法注册表
# ═══════════════════════════════════════════════════════════════════════

METHODS = {
    # MACE family
    'mace_polar': compute_mace_polar,
    'mace_omol':  compute_mace_omol,
    'mace_off23': compute_mace_off23,
    'mace_off24': compute_mace_off24,
    # AIMNet
    'aimnet2': compute_aimnet2,
    # DeepMD family
    'dpa3':      compute_dpa3,
    'dpa2_drug': compute_dpa2_drug,
    # Orb
    'orbmol':    compute_orbmol,
    'orbmol_v2': compute_orbmol_v2,
    # xTB family
    'xtb':  compute_xtb,
    'gxtb': compute_gxtb,
    # ANI family
    'ani_1x':   compute_ani_1x,
    'ani_1ccx': compute_ani_1ccx,
    'ani_2x':   compute_ani_2x,
    # MLatom CLI
    'd4ani': compute_d4ani,
    'aiqm3': compute_aiqm3,
    # Other
    'fennix': compute_fennix,
    'nutmeg': compute_nutmeg,
    'pm6ml':  compute_pm6ml,
    'aceff2': compute_aceff2,
}


def compute_energy(filepath, method, charge=None, spin=None):
    """使用指定 MLIP 方法计算 XYZ 文件的能量。

    Args:
        filepath: XYZ 文件路径
        method:   MLIP 方法名 (见 METHODS)
        charge:   体系总电荷（None 则从 XYZ 第二行读取）
        spin:     自旋多重度（None 则从 XYZ 第二行读取）

    Returns:
        能量 (eV)
    """
    import time
    from ase.io import read

    atoms = read(filepath)

    # 从 xyz 文件第二行读取 charge 和 spin（如果未显式指定）
    if charge is None or spin is None:
        with open(filepath, 'r') as f:
            next(f)
            parts = next(f).rstrip('\n').split()
            if charge is None:
                charge = int(parts[0])
            if spin is None:
                spin = int(parts[1])

    start = time.time()

    if method not in METHODS:
        available = ', '.join(sorted(METHODS.keys()))
        raise ValueError(f"Unknown method '{method}'. Available: {available}")

    ene = METHODS[method](atoms, charge, spin)
    if isinstance(ene, (list, np.ndarray)):
        ene = ene[0] if hasattr(ene, '__getitem__') and len(ene) > 0 else float(ene)

    elapsed = time.time() - start
    print(f'Energy: {ene / EV_HA:.12f} Hartree')
    print(f'Elapsed: {elapsed:.4f} s')
    return ene


# ═══════════════════════════════════════════════════════════════════════
#  CLI 入口 (兼容 MLIP_xyz.py 用法)
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print(__doc__)
        print('Usage: python calculators.py <xyz_file> <method>')
        print(f'Available methods: {", ".join(sorted(METHODS.keys()))}')
        sys.exit(1)

    compute_energy(sys.argv[1], sys.argv[2])
