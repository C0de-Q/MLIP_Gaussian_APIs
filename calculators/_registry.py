"""
calculators/_registry.py — registry, registration API, declarative ASE support

The model modules (_ase_models / _aimnet / _orb / _xtb / _ani / _cli) only define
models and register them through the register_* functions; this file contains no
concrete model.
"""

import threading
import time
import logging
import numpy as np
from constants import require_model_path, server_device
import log_utils  # importing it configures the root logger

logger = logging.getLogger('calculators')


# ═══════════════════════════════════════════════════════════════════════
#  Registry
# ═══════════════════════════════════════════════════════════════════════

METHODS = {}
GRADIENTS = {}
ASE_MODEL_SPECS = {}

# In-process model cache: functional models load once per server process.
_MODEL_CACHE = {}
_MODEL_CACHE_LOCK = threading.Lock()

# Model-loading time of the request being served, per thread. Loading happens
# inside the first request of a lazily loaded model, so the server subtracts this
# from the step time it reports and shows it as a separate field.
_LOAD_TIME = threading.local()


def take_load_time():
    """Return the model-loading seconds recorded in this thread, and reset it."""
    seconds = getattr(_LOAD_TIME, 'seconds', 0.0)
    _LOAD_TIME.seconds = 0.0
    return seconds


def get_cached_model(key, builder):
    """Cache a model or calculator under key; builder runs only once.

    Functional models use this so that repeated requests in one process (for
    example inside mlip_server) do not reload anything. It does not suit CLI
    models, which run an external program per request, nor objects built from the
    specific geometry (an xtb Calculator carries its coordinates); handle those
    cases inside the compute function instead.
    """
    with _MODEL_CACHE_LOCK:
        obj = _MODEL_CACHE.get(key)
        if obj is None:
            t0 = time.perf_counter()
            obj = builder()
            elapsed = time.perf_counter() - t0
            _LOAD_TIME.seconds = getattr(_LOAD_TIME, 'seconds', 0.0) + elapsed
            logger.info('Model loaded: key=%s in %.2fs', key, elapsed)
            _MODEL_CACHE[key] = obj
        return obj


def register_method(name, fn=None):
    """Register a custom energy function, as a decorator or a direct call:

        @register_method('name')
        def compute_name(...): ...

        register_method('name', compute_name)
    """
    def _reg(f):
        METHODS[name] = f
        return f
    return _reg(fn) if fn is not None else _reg


def register_gradient(name, fn=None):
    """Register a custom gradient function (used like register_method)."""
    def _reg(f):
        GRADIENTS[name] = f
        return f
    return _reg(fn) if fn is not None else _reg


# ═══════════════════════════════════════════════════════════════════════
#  Declarative ASE models
# ═══════════════════════════════════════════════════════════════════════

def make_ase_calculator(spec, device=None):
    """Build an ASE calculator from a declarative spec.

    The server's persistent ASE path calls this once at startup and reuses the
    result for every later request. It accepts the same spec keys as
    _make_ase_compute (module/factory/model/device/default_dtype/factory_kwargs).

    The device argument wins, then the spec's device key, and a device is only
    passed on when one of them is set. The spec key accepts:
        True             the factory takes a device; use MLIP_SERVER_DEVICE
        'cpu', 'cuda:0'  an explicit device, overriding the configuration
        missing          call the factory without a device argument
    """
    mod = __import__(spec['module'], fromlist=[spec['factory']])
    factory = getattr(mod, spec['factory'])

    kwargs = dict(spec.get('factory_kwargs', {}))
    if 'model' in spec:
        kwargs['model'] = require_model_path(spec['model'])
    if device is None:
        device = spec.get('device')
        if device is True:
            device = server_device()
    if device:
        kwargs['device'] = device
    if spec.get('default_dtype'):
        kwargs['default_dtype'] = spec['default_dtype']
    return factory(**kwargs)


def prepare_ase_atoms(atoms, charge, spin, spec):
    """Write charge/spin and model-specific fields into the Atoms (per step)."""
    if spec.get('needs_info', True):
        atoms.info['charge'] = int(charge)
        atoms.info['spin'] = int(spin)
    prepare = spec.get('prepare')
    if prepare is not None:
        prepare(atoms, charge, spin)


def _make_ase_compute(spec):
    """Build a standard ASE calculator energy function from a spec (returns eV).

    Supported spec keys:
        module         required, Python package or module name
        factory        required, calculator class or factory function
        model          model key; require_model_path resolves the weight path
        device         passed to factory as device (omitted by default)
        default_dtype  precision argument required by MACE and similar models
        needs_info     whether to write atoms.info['charge'/'spin'] (default True)
        prepare        callable(atoms, charge, spin) run before each evaluation;
                       model-specific fields belong here in each model file's
                       spec, for example DPA3's fparam or DPA4's charge_spin
        factory_kwargs extra keyword arguments passed to factory
    """
    def compute(atoms, charge, spin, base=None):
        prepare_ase_atoms(atoms, charge, spin, spec)
        atoms.calc = make_ase_calculator(spec)
        return atoms.get_potential_energy()  # eV

    compute.__name__ = 'compute_' + spec.get('name', spec['factory'])
    return compute


def register_ase_model(name, spec):
    """Register a standard ASE calculator model in one line."""
    METHODS[name] = _make_ase_compute(spec)
    ASE_MODEL_SPECS[name] = spec


def get_ase_spec(name):
    """Return the spec of a declarative ASE model, or None for other models."""
    return ASE_MODEL_SPECS.get(name)


def _default_gradient(atoms, charge, spin, base=None):
    """Default gradient: ASE forces are eV/Å, so dE/dr is their negation."""
    forces = atoms.get_forces()
    return [[-f[0], -f[1], -f[2]] for f in forces]


# ═══════════════════════════════════════════════════════════════════════
#  Unified entry points
# ═══════════════════════════════════════════════════════════════════════

def compute(method, atoms, charge, spin, base=None):
    """Energy entry point: call the energy function for a method, in eV."""
    if method not in METHODS:
        raise ValueError(
            f"Unknown method '{method}'. Available: {', '.join(sorted(METHODS))}")
    fn = METHODS[method]
    try:
        return fn(atoms, charge, spin, base=base)
    except TypeError:
        return fn(atoms, charge, spin)


def compute_gradient(method, atoms, charge, spin, base=None):
    """Gradient entry point: dE/dr in eV/Å; unregistered methods use the ASE default."""
    fn = GRADIENTS.get(method, _default_gradient)
    try:
        return fn(atoms, charge, spin, base=base)
    except TypeError:
        return fn(atoms, charge, spin)
