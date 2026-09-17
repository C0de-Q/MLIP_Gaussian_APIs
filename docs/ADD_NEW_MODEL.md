# Adding a new MLIP interface

Model loading lives in the server, so adding a model never means editing the
server. Copy `scripts/custom_models.example.py` to `custom_models.py` in the
repository root and register the model there; the `calculators` package loads that
file automatically when it is imported.

Three steps:

1. **Register the model** — Option A, B or C below.
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

## Option B: functional, gradient carried on the atoms

A decorated function whose energy call leaves the forces on the atoms, so the
gradient function is a lookup instead of a second evaluation:

```python
from calculators import register_method, register_gradient, store_result, take_result

@register_method('mymodel2')
def compute_mymodel2(atoms, charge, spin, base=None):
    atoms.calc = MyCalculator(model=require_model_path('mymodel2'))   # or
    # atoms.arrays['forces'] = ...      # if the model hands you forces directly
    return atoms.get_potential_energy()  # eV

@register_gradient('mymodel2')
def grad_mymodel2(atoms, charge, spin, base=None):
    forces = atoms.get_forces()          # already computed by the energy call
    return [[-f[0], -f[1], -f[2]] for f in forces]   # dE/dr in eV/Å
```

That is the server's default ASE gradient written out. With `atoms.calc` set you may
leave the gradient function out, but writing it keeps the sign and units visible;
when the energy call only fills `atoms.arrays['forces']`, it is required — there is
no calculator for `atoms.get_forces()` to use.

The model object itself lives outside the step: `get_cached_model(key, builder)`
builds it once per process, so a slow model is not reloaded on every request.

## Option C: functional, gradient handed over by `base`

For a gradient that cannot ride on the atoms — autograd over your own tensors, a CLI
that returns energy and gradient together, any non-ASE interface — pass the result
from the energy call to the gradient call through `base`, which both calls receive
and which is unique per request:

```python
@register_method('mymodel2')
def compute_mymodel2(atoms, charge, spin, base=None):
    energy, grad = my_model(atoms)       # one evaluation, both results
    store_result(base, energy=energy, grad=grad)
    return energy                        # eV

@register_gradient('mymodel2')
def grad_mymodel2(atoms, charge, spin, base=None):
    cached = take_result(base)
    if cached is not None:
        return cached['grad']            # dE/dr in eV/Å, shape (natoms, 3)
    return my_model(atoms)[1]            # no cache: compute it here
```

`store_result` and `take_result` come from `calculators/`: a dict keyed by `base`,
locked and bounded, so the two calls of one request see the same values and nothing
accumulates. Keep the fallback branch — a gradient request must work even without a
preceding energy call — and cache the model itself with `get_cached_model` as in
Option B.

Working examples: `calculators/_aimnet.py` (Option B, calculator attached plus the
explicit gradient), `calculators/_orb.py` `orbmol_v2` (Option B, forces on the
atoms), `calculators/_ani.py` and `calculators/_cli.py` (Option C, one evaluation
then handed over through `base`).

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
persisted automatically; Options B and C are loaded on the first request only if
the function caches them with `get_cached_model`. RunMLIPgjf.sh sends one warm-up
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
