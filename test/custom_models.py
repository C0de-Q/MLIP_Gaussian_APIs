"""test/custom_models.py — declarative ASE model plugin used by the tests

This is written exactly like the custom_models.py a user would create (see
scripts/custom_models.example.py). The difference is that this file is the
subject under test: test/test_add_ase_model.py loads it explicitly, whereas the
server only loads custom_models.py from the repository root.

Two models are registered:

    emttest  the EMT calculator that ships with ASE, so the whole chain can be
             tested without weights or network access
    dpa4     the real DeepMD DPA4 registration, including the prepare hook that
             writes charge_spin before each evaluation; equivalent to the
             built-in dpa4 in calculators/_ase_models.py, and it needs deepmd-kit
             plus MLIP_MODEL_DPA4 weights to actually compute

Declarative registration means the server started by RunMLIPgjf.sh takes the
persistent ASE path: the calculator is built once and reused for every
optimization step.
"""

import numpy as np

ASE_CALCULATORS_EXTRA = {
    # Model under test: a standard ASE calculator needs only this entry, and
    # gradients come from the ASE forces.
    'emttest': dict(
        module='ase.calculators.emt',
        factory='EMT',
    ),
    # Real-model form: a standard ASE calculator plus a prepare hook for the
    # fields that model needs.
    'dpa4': dict(
        module='deepmd.calculator',
        factory='DP',
        model='dpa4',                 # read from MLIP_MODEL_DPA4 in config.env
        prepare=lambda atoms, charge, spin: atoms.info.update(
            {'charge_spin': np.array([charge, spin])}),
    ),
}
