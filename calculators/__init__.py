"""
calculators — energy and gradient calculators for every supported model

Split by model family:
    _registry.py   registry, registration API, declarative ASE support, entry points
    _ase_models.py built-in declarative ASE models (MACE, DPA)
    _aimnet.py     AIMNet2
    _orb.py        ORB-Mol v3 / v2
    _xtb.py        GFN2-xTB (Python API and gxtb)
    _ani.py        ANI-1x / 1ccx / 2x
    _cli.py        MLatom/aitomic CLI models (d4ani, aiqm3)

Every energy calculator has the signature
    def compute_xxx(atoms, charge, spin, base=None) -> float   # returns eV
and every gradient calculator
    def grad_xxx(atoms, charge, spin, base=None) -> list[natoms][3] | None
    # returns dE/dr in eV/Å

There are three ways to add a model, from simplest to most involved:
  1. Declarative (recommended for a standard ASE calculator) — one entry in
     custom_models.py in the repository root:
          ASE_CALCULATORS_EXTRA = {
              'mymodel': dict(module='mypkg', factory='MyCalc',
                              model='mymodel', device=True),
          }
     No changes to this package, and the server uses the persistent ASE path.
  2. Functional decorators, also in custom_models.py:
      @register_method('mymodel')
      def compute_mymodel(atoms, charge, spin, base=None): ...
  3. A new module inside this package.

See docs/ADD_NEW_MODEL.md and scripts/custom_models.example.py.
"""

import importlib.util
import logging
import os
import sys

from ._registry import (
    METHODS,
    GRADIENTS,
    ASE_MODEL_SPECS,
    register_method,
    register_gradient,
    get_cached_model,
    take_load_time,
    store_result,
    take_result,
    make_ase_calculator,
    prepare_ase_atoms,
    register_ase_model,
    get_ase_spec,
    compute,
    compute_gradient,
)

# Importing a model module registers it.
from . import _ase_models, _aimnet, _orb, _xtb, _ani, _cli
from ._ase_models import BUILTIN_ASE_MODELS

__all__ = [
    'METHODS', 'GRADIENTS', 'ASE_MODEL_SPECS', 'BUILTIN_ASE_MODELS',
    'register_method', 'register_gradient', 'register_ase_model',
    'get_cached_model', 'make_ase_calculator', 'prepare_ase_atoms', 'get_ase_spec',
    'compute', 'compute_gradient', 'load_custom_models', 'take_load_time',
    'store_result', 'take_result',
]

logger = logging.getLogger('calculators')


def load_custom_models(path=None):
    """Load custom_models.py from the repository root (optional user plugin).

    This is the entry point behind "Option A/B" in README and
    docs/ADD_NEW_MODEL.md: copy scripts/custom_models.example.py to
    custom_models.py in the repository root; no changes to this package needed.

    Two kinds of content are supported:
      - ASE_CALCULATORS_EXTRA: dict, declarative registration of ASE models
      - register_method / register_gradient decorators, which run on import

    The file is loaded by path through importlib, so the repository root does not
    have to be on sys.path. A missing file is not an error: most users have no
    custom models.
    """
    if path is None:
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'custom_models.py')
    if not os.path.isfile(path):
        return None

    spec = importlib.util.spec_from_file_location('mlip_custom_models', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['mlip_custom_models'] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop('mlip_custom_models', None)
        raise

    extra = dict(getattr(module, 'ASE_CALCULATORS_EXTRA', {}))
    for name, model_spec in extra.items():
        register_ase_model(name, model_spec)
    logger.info('Loaded custom model plugin %s (declarative: %s)', path,
                ', '.join(extra) if extra else 'none, decorator-based')
    return module


load_custom_models()
