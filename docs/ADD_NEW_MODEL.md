# Adding a new MLIP interface

Model loading lives in the server, so adding a model never means editing the
server. Copy `scripts/custom_models.example.py` to `custom_models.py` in the
repository root and register the model there; the `calculators` package loads that
file automatically when it is imported.

Three steps:

1. **Register the model** — Option A or B below.
2. **Add its weight path to `config.env`** if it loads a weight file:

   ```bash
   MLIP_MODEL_MYMODEL=/abs/path/to/weights.file
   ```

   The name follows the model key: `model='mymodel'` or
   `require_model_path('mymodel')` reads `MLIP_MODEL_MYMODEL`. It must be a full
   path including the file name; a blank value fails with an error naming the
   variable. Models that fetch their own weights (`orbmol`) need no entry, and an
   environment variable of the same name wins over `config.env`.
3. **Run it**: `./RunMLIPgjf.sh -m mymodel job.gjf`, with `external=` in the job
   pointing at `gau_scripts/Gau_generic.py`. Optional: run
   `python scripts/gen_gau_scripts.py` to write `Gau_<model>.py` wrappers that
   also check the model name.

## Option A: declarative (recommended for standard ASE calculators)

```python
ASE_CALCULATORS_EXTRA = {
    'mymodel': dict(
        module='mypackage',      # package providing the calculator
        factory='MyCalculator',  # calculator class or factory function
        model='mymodel',         # weights from MLIP_MODEL_MYMODEL
        device=True,             # factory takes a device; use MLIP_SERVER_DEVICE
        # default_dtype='float32', factory_kwargs={'precision': 'high'},
    ),
}
```

The calculator is built once when the server starts and reused for every step,
which satisfies the "the model must be loaded up front" rule without extra code.
Leave out `model` when the calculator needs no weight file.

Spec keys: `module`, `factory`, `model`, `device`, `default_dtype`, `needs_info`,
`prepare`, `factory_kwargs`.

- `device=True` passes `MLIP_SERVER_DEVICE`; a string such as `'cpu'` pins one
  model; leaving the key out calls the factory without a device.
- `prepare` is a `callable(atoms, charge, spin)` run before every step, for fields
  a calculator expects elsewhere: DPA3 uses `atoms.info['fparam']`, DPA4
  `atoms.info['charge_spin']`, MACE-POLAR also needs `external_field`.
- By default the job's charge and spin multiplicity arrive in
  `atoms.info['charge']` and `['spin']`, which is exactly what MACE reads; use
  `needs_info=False` to skip them.

## Option B: functional (autograd, CLI, non-ASE interfaces)

Decorated functions in `custom_models.py`; anything that returns eV and dE/dr in
eV/Å can be hosted. They are **not** persisted by default, so a model that is
expensive to build must be cached inside the function:

```python
@register_method('mymodel2')
def compute_mymodel2(atoms, charge, spin, base=None):
    return energy_ev            # eV

@register_gradient('mymodel2')  # only if the default ASE gradient does not fit
def grad_mymodel2(atoms, charge, spin, base=None):
    return grad                 # dE/dr in eV/Å, shape (natoms, 3)
```

For a slow model, build it once and reuse the instance:

```python
def _build():
    from constants import require_model_path, server_device
    from mypackage import MyModel
    return MyModel(model=require_model_path('mymodel2'), device=server_device())

@register_method('mymodel2')
def compute_mymodel2(atoms, charge, spin, base=None):
    model = get_cached_model('mymodel2_model', _build)   # loaded once
    ...

@register_gradient('mymodel2')
def grad_mymodel2(atoms, charge, spin, base=None):
    model = get_cached_model('mymodel2_model', _build)   # same instance
    ...
```

Working examples: `calculators/_ani.py` (autograd, cached), `calculators/_orb.py`
(non-ASE model plus adapter), `calculators/_cli.py` (external program in a
temporary directory, so it cannot be cached).

## Gradients and forces

A gradient and a force differ by a sign, and a wrong sign does not fail loudly — it
relaxes the geometry the wrong way.

- `compute_*` returns eV; `grad_*` returns **dE/dr in eV/Å**, shape `(natoms, 3)`.
- `atoms.get_forces()` is the force `-dE/dr`, so negate it:
  `grad = [[-f[0], -f[1], -f[2]] for f in forces]`.
- Most MLIP packages return forces, so check the sign in their documentation.
- Convert units yourself (torchani and MLatom report Hartree and Ha/Å; use
  `EV_HA` and `FORCE_UNIT_CONST`); the server then converts to Gaussian's units.
- Never return zeros or `None` when no gradient is available: the server writes
  zero forces, which Gaussian reads as a stationary structure. Raise instead, and
  the job stops.

## How the server loads it

Nothing else to do — a declarative entry or `@register_method` is enough for the
generic path, and `MLIPServer` makes no method-name decisions. Option A is
persisted automatically; Option B is loaded on the first request only if the
function caches it with `get_cached_model`. RunMLIPgjf.sh sends one warm-up
calculation before the jobs, so that first load happens there and not inside your
first Gaussian step.

## Testing your model

[`../test/`](../test/) registers a model, checks registration and persistence, and
compares energy and gradients against a direct ASE calculation without weights:

```bash
pytest test/ -v
```

Use `test/custom_models.py` as a template for your own plugin. Two habits keep
models usable from Gaussian: name scratch files from `unique_scratch_base()` and
remove them again, and never hard-code a weight path (`require_model_path('key')`).
