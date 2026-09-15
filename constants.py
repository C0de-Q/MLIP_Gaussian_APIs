#!/usr/bin/env python3
"""
constants.py — shared configuration: unit constants, element table, model and
program paths

Used by the calculators package, gaussian_external.py, gau_scripts/*,
mlip_server.py, and others so that shared values are defined once.

Path configuration (the repository ships no machine-specific paths):
    Model files and external programs are supplied by the user, in one of two
    ways:
      1. Copy config.example.env to config.env and fill it in (recommended).
      2. Export environment variables with the same names (takes precedence).

    Model paths must include the file name; the directory is never searched:
      MLIP_MODEL_<KEY>=/abs/path/to/model-file
      e.g. MLIP_MODEL_MACE_OMOL=/data/models/MACE-omol-extra-large.model

    A missing path raises a clear error at the point where that model is used.
"""

import os

# ═══════════════════════════════════════════════════════════════════════
#  Unit conversion constants
# ═══════════════════════════════════════════════════════════════════════
EV_HA = 27.211386245              # eV → Hartree
BOHR_TO_ANGSTROM = 0.52917721067  # Bohr → Å
FORCE_UNIT_CONST = EV_HA / BOHR_TO_ANGSTROM   # Hartree/Bohr ↔ eV/Å

# ═══════════════════════════════════════════════════════════════════════
#  Element table (1-indexed, index 0 is a placeholder, up to Rn)
# ═══════════════════════════════════════════════════════════════════════
ELEMENTS = [
    None,   # placeholder so atomic numbers can be used as indices directly
    'H',  'He', 'Li', 'Be', 'B',  'C',  'N',  'O',  'F',  'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P',  'S',  'Cl', 'Ar',
    'K',  'Ca', 'Sc', 'Ti', 'V',  'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y',  'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I',  'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf',
    'Ta', 'W',  'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po',
    'At', 'Rn',
]

# ═══════════════════════════════════════════════════════════════════════
#  Configuration file loading (config.env in the repository root)
# ═══════════════════════════════════════════════════════════════════════
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _load_dotenv(path=None):
    """Minimal .env parser: KEY=VALUE, supports # comments and quotes.

    Existing environment variables take precedence.
    """
    if path is None:
        path = os.path.join(_REPO_ROOT, 'config.env')
    if not os.path.isfile(path):
        return
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, value = line.partition('=')
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)


_load_dotenv()

# ═══════════════════════════════════════════════════════════════════════
#  Model file paths (full path including file name, no directory search)
# ═══════════════════════════════════════════════════════════════════════


def require_model_path(key):
    """Return the full model path, or raise an error explaining what to set."""
    env_name = 'MLIP_MODEL_' + key.upper()
    path = os.environ.get(env_name)
    if not path:
        raise RuntimeError(
            f'Missing model path for "{key}".\n'
            f'  Set MLIP_MODEL_{key.upper()} to a full path including the file name,\n'
            f'  for example MLIP_MODEL_{key.upper()}=/data/models/xxx.model,\n'
            f'  or copy config.example.env to config.env in the repository root\n'
            f'  and fill it in.'
        )
    return path


# ═══════════════════════════════════════════════════════════════════════
#  Executable paths for external programs (set them, or keep them on PATH)
# ═══════════════════════════════════════════════════════════════════════
CLI_PATHS = {
    'mlatom':  os.environ.get('MLIP_MLATOM_BIN', 'mlatom'),
    'aitomic': os.environ.get('MLIP_AITOMIC_BIN', 'aitomic'),
    'xtb':     os.environ.get('MLIP_XTB_BIN', 'xtb'),
}


def require_cli_path(key):
    """Return the configured program path, or raise an error explaining what to set."""
    path = CLI_PATHS.get(key)
    if not path:
        raise RuntimeError(
            f'Missing path for the external program "{key}".\n'
            f'  Set MLIP_{key.upper()}_BIN (or fill it in config.env).'
        )
    return path


# ═══════════════════════════════════════════════════════════════════════
#  Other runtime settings (all overridable through environment variables)
# ═══════════════════════════════════════════════════════════════════════


def server_device():
    """Return the device used for model evaluation.

    Every model the server runs uses this single setting: the declarative ASE
    calculators and the cached functional ones such as mace_polar or orbmol_v2.
    MLIP_SERVER_DEVICE wins; without it the device is cuda when torch reports an
    available GPU and cpu otherwise. Anything torch.device accepts is valid, for
    example 'cpu' or 'cuda:0'.
    """
    device = os.environ.get('MLIP_SERVER_DEVICE')
    if device:
        return device
    try:
        import torch
    except ImportError:
        return 'cpu'
    return 'cuda' if torch.cuda.is_available() else 'cpu'
