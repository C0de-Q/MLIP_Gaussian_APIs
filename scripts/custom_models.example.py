#!/usr/bin/env python3
"""
custom_models.py — template for your own model plugins

Copy this file to custom_models.py in the repository root and fill it in. The
calculators package reads it automatically when it is imported (which includes
starting the server through RunMLIPgjf.sh), so adding a model never requires
changes to the calculators package or constants.py. Your custom_models.py is
git-ignored and stays local.

Remember step 2 of docs/ADD_NEW_MODEL.md: a model that loads a weight file also
needs its path in config.env (MLIP_MODEL_<KEY>=/abs/path/with/file.name), and the
key is the one you use below — model='mymodel' or require_model_path('mymodel')
both read MLIP_MODEL_MYMODEL.

Two styles, which may be combined:
  1. Declarative (recommended for a standard ASE calculator): fill in
     ASE_CALCULATORS_EXTRA and gradients come from the ASE forces. The server
     started by RunMLIPgjf.sh builds the calculator once and reuses it (the
     persistent ASE path), with no extra code.
  2. Functional (for autograd, CLI, or non-ASE interfaces): use the
     @register_method / @register_gradient decorators. Note that functional
     models are rebuilt on every step and are not persisted automatically, so a
     model that is expensive to construct — autograd models in particular — must
     be cached with get_cached_model inside the compute function (see 2b below,
     and "Option B" in docs/ADD_NEW_MODEL.md).

Then run:
    ./RunMLIPgjf.sh -m mymodel job.gjf
(with the Gaussian external pointing at gau_scripts/Gau_generic.py or Gau_mymodel.py)
"""

from calculators import register_method, register_gradient, register_ase_model


# ── 1) Declarative: standard ASE calculators ────────────────────────────
ASE_CALCULATORS_EXTRA = {
    # 'mymodel': dict(
    #     module='mypackage',          # Python package providing the calculator
    #     factory='MyCalculator',      # calculator class or factory function
    #     model='mymodel',             # read from MLIP_MODEL_MYMODEL (full path with file name)
    #     device=True,                 # the factory takes a device; use MLIP_SERVER_DEVICE
    #     # default_dtype='float32',
    #     # factory_kwargs={'precision': 'high'},
    # ),
    #
    # Add a prepare hook when the model needs extra fields in atoms.info; the
    # built-in dpa3 and dpa4 models are written that way (see
    # calculators/_ase_models.py):
    # 'mymodel_with_field': dict(
    #     module='deepmd.calculator',
    #     factory='DP',
    #     model='mymodel_with_field',  # read from MLIP_MODEL_MYMODEL_WITH_FIELD
    #     prepare=lambda atoms, charge, spin: atoms.info.update(
    #         {'charge_spin': np.array([charge, spin])}),
    # ),
}


# ── 2) Functional: any model ────────────────────────────────────────────
# 2a. Simple form: rebuilt on every step, so only for cheap-to-build models.
# @register_method('mymodel2')
# def compute_mymodel2(atoms, charge, spin, base=None):
#     """Return the energy in eV."""
#     from mypackage import MyCalculator
#     from constants import require_model_path, server_device
#     atoms.info["charge"] = int(charge)
#     atoms.info["spin"] = int(spin)
#     atoms.calc = MyCalculator(model=require_model_path('mymodel2'),
#                               device=server_device())
#     return atoms.get_potential_energy()  # eV


# @register_gradient('mymodel2')   # only if the default ASE gradient does not apply
# def grad_mymodel2(atoms, charge, spin, base=None):
#     """Return dE/dr in eV/Å with shape (natoms, 3)."""
#     forces = atoms.get_forces()  # ASE forces in eV/Å, i.e. -dE/dr
#     return [[-f[0], -f[1], -f[2]] for f in forces]


# 2b. Persistent form (recommended; autograd models must cache the model with
#     get_cached_model so that each step only builds the input tensors):
# from calculators import get_cached_model
# from constants import EV_HA
#
# def _build_mymodel2():
#     """Runs once: load the weights and move the model to the device."""
#     from mypackage import MyModel
#     from constants import require_model_path, server_device
#     return MyModel(model=require_model_path('mymodel2'), device=server_device())
#
# @register_method('mymodel2')
# def compute_mymodel2_persistent(atoms, charge, spin, base=None):
#     model = get_cached_model('mymodel2_model', _build_mymodel2)
#     ...
#     return energy_ev  # eV
#
# @register_gradient('mymodel2')
# def grad_mymodel2_persistent(atoms, charge, spin, base=None):
#     model = get_cached_model('mymodel2_model', _build_mymodel2)  # shared with the energy
#     ...
#     return grad  # dE/dr in eV/Å
