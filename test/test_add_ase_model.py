#!/usr/bin/env python3
"""Test cases for adding an ASE model (declarative registration).

These cover the full chain described in the "Adding a model" section of the
README and in "Option A" of docs/ADD_NEW_MODEL.md. Apart from the DPA4 reference
case, nothing here needs model weights or network access: the model under test is
ASE's built-in EMT calculator, registered declaratively in test/custom_models.py.

Run with:

    pytest test/ -v

The chain being verified:

    test/custom_models.py            declarative plugin (emttest and dpa4)
    calculators.load_custom_models   loads the plugin by path and registers it
    MLIPServer                       builds the calculator once, then reuses it
    gau_scripts/Gau_*.py             Gaussian External forwarding, including the
                                     model name check
"""

import os
import pathlib
import socket
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TEST_DIR = pathlib.Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import calculators  # noqa: E402
from constants import BOHR_TO_ANGSTROM, EV_HA, FORCE_UNIT_CONST  # noqa: E402

TEST_MODEL = 'emttest'          # the weight-free model registered in test/custom_models.py


# ═══════════════════════════════════════════════════════════════════════
#  Gaussian External input/output helpers and port utilities
# ═══════════════════════════════════════════════════════════════════════

def make_water():
    """Return a small molecule from ASE, so no structure file is needed."""
    from ase.build import molecule
    return molecule('H2O')


def write_external_input(path, atoms, deriva=1, charge=0, spin=1):
    """Write an input file in Gaussian External format (Bohr, 5th column point charge)."""
    lines = [f'{len(atoms)} {deriva} {charge} {spin}']
    for z, p in zip(atoms.get_atomic_numbers(), atoms.get_positions()):
        lines.append(
            f'{int(z):4d}'
            f'{p[0] / BOHR_TO_ANGSTROM:20.12f}'
            f'{p[1] / BOHR_TO_ANGSTROM:20.12f}'
            f'{p[2] / BOHR_TO_ANGSTROM:20.12f}{0.0:12.6f}')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def read_external_output(path):
    """Read a Gaussian External output file: energy in Ha and gradient in Ha/Bohr."""
    rows = [r for r in path.read_text(encoding='utf-8').splitlines() if r.strip()]
    energy_ha = float(rows[0][:20])
    grad = np.array([[float(r[20 * i:20 * i + 20]) for i in range(3)]
                     for r in rows[1:]])
    return energy_ha, grad


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def wait_port(port, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(('127.0.0.1', port), 1):
                return True
        except OSError:
            time.sleep(0.1)
    return False


# ═══════════════════════════════════════════════════════════════════════
#  fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture(scope='module')
def plugin():
    """Load test/custom_models.py and restore the registry afterwards."""
    methods, specs = dict(calculators.METHODS), dict(calculators.ASE_MODEL_SPECS)
    module = calculators.load_custom_models(str(TEST_DIR / 'custom_models.py'))
    assert module is not None, 'test/custom_models.py was not loaded'
    yield module
    calculators.METHODS.clear()
    calculators.METHODS.update(methods)
    calculators.ASE_MODEL_SPECS.clear()
    calculators.ASE_MODEL_SPECS.update(specs)


@pytest.fixture(scope='module')
def server(plugin):
    """Start a persistent server the way RunMLIPgjf.sh does for a declarative model."""
    from mlip_server import MLIPServer

    port = free_port()
    srv = MLIPServer(method=TEST_MODEL, host='127.0.0.1', port=port)
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    assert wait_port(port), 'the MLIP server did not become ready in time'
    yield srv, port
    srv.shutdown()
    thread.join(timeout=15)


@pytest.fixture
def job(tmp_path):
    """Input and output file paths for one Gaussian External request."""
    return tmp_path / 'input.dat', tmp_path / 'output.dat'


# ═══════════════════════════════════════════════════════════════════════
#  Option A: declarative registration
# ═══════════════════════════════════════════════════════════════════════

def test_plugin_registers_models(plugin):
    """A single ASE_CALCULATORS_EXTRA entry registers a model, with no library changes."""
    assert TEST_MODEL in calculators.METHODS
    spec = calculators.get_ase_spec(TEST_MODEL)
    assert spec is not None, 'a declarative model must appear in ASE_MODEL_SPECS to be persisted'
    assert (spec['module'], spec['factory']) == ('ase.calculators.emt', 'EMT')

    # The real-model form registers just as well (the spec is lazy: it neither
    # imports deepmd nor reads a weight file).
    dpa4 = calculators.get_ase_spec('dpa4')
    assert dpa4 is not None and dpa4['model'] == 'dpa4'
    assert callable(dpa4['prepare']), 'dpa4 needs a prepare hook to write charge_spin'


def test_spec_device_comes_from_configuration(tmp_path, monkeypatch):
    """device=True means "use MLIP_SERVER_DEVICE", so no spec hard-codes cpu."""
    (tmp_path / 'deviceprobe.py').write_text(
        'class Probe:\n'
        '    def __init__(self, **kwargs):\n'
        '        self.kwargs = kwargs\n',
        encoding='utf-8')
    monkeypatch.syspath_prepend(str(tmp_path))

    monkeypatch.setenv('MLIP_SERVER_DEVICE', 'cuda:7')
    calc = calculators.make_ase_calculator(
        {'module': 'deviceprobe', 'factory': 'Probe', 'device': True})
    assert calc.kwargs['device'] == 'cuda:7'

    # A string still pins one model explicitly.
    calc = calculators.make_ase_calculator(
        {'module': 'deviceprobe', 'factory': 'Probe', 'device': 'cpu'})
    assert calc.kwargs['device'] == 'cpu'

    # Without the key the factory is called with no device argument at all.
    calc = calculators.make_ase_calculator(
        {'module': 'deviceprobe', 'factory': 'Probe'})
    assert 'device' not in calc.kwargs


def test_load_time_is_reported_separately():
    """The model load is measured apart from the step it happens in."""
    calculators.take_load_time()          # start from a clean slate

    def _build():
        time.sleep(0.05)
        return object()

    calculators.get_cached_model('loadtime-test', _build)
    first = calculators.take_load_time()
    assert first >= 0.05, f'the build time should be recorded, got {first}'
    assert calculators.take_load_time() == 0.0, 'reading it clears the counter'

    calculators.get_cached_model('loadtime-test', _build)   # cached now
    assert calculators.take_load_time() == 0.0, 'a cache hit adds no load time'


def test_result_hand_over_is_bounded_and_cleared():
    """store_result/take_result pass values between the two entry points."""
    calculators.store_result('handover-a', energy=-1.0, grad=[[0.0, 0.0, 0.0]])
    stored = calculators.take_result('handover-a')
    assert stored['energy'] == -1.0 and stored['grad'] == [[0.0, 0.0, 0.0]]
    assert calculators.take_result('handover-a') is None, 'taking clears it'

    calculators.store_result(None, energy=1.0)          # no base: nothing stored
    assert calculators.take_result(None) is None

    for i in range(12):                                 # the cache stays small
        calculators.store_result(f'handover-{i}', energy=float(i))
    assert calculators.take_result('handover-0') is None, 'oldest entry evicted'
    assert calculators.take_result('handover-11')['energy'] == 11.0


def test_server_builds_calculator_once(server, job):
    """Persistent ASE path: the calculator is built at startup and reused per step."""
    srv, _port = server
    assert srv.ase_calc is not None, 'a declarative model should build its calculator at startup'
    calculator_at_startup = srv.ase_calc

    for tag, shift in (('step1', 0.0), ('step2', 0.01)):
        atoms = make_water()
        atoms.positions += shift
        inp = job[0].with_name(f'{tag}_in.dat')
        out = job[1].with_name(f'{tag}_out.dat')
        write_external_input(inp, atoms)
        srv.compute(str(inp), str(out))

    assert srv.ase_calc is calculator_at_startup, 'the second step must not reload the model'


def test_energy_and_gradient_match_ase(server, job):
    """Energy and gradient match a direct ASE calculation, including unit conversion."""
    from ase.calculators.emt import EMT

    srv, _port = server
    inp, out = job
    atoms = make_water()
    write_external_input(inp, atoms)
    srv.compute(str(inp), str(out))
    energy_ha, grad = read_external_output(out)

    ref = make_water()
    ref.calc = EMT()
    assert energy_ha == pytest.approx(ref.get_potential_energy() / EV_HA, abs=1e-10)
    np.testing.assert_allclose(grad, -ref.get_forces() / FORCE_UNIT_CONST,
                               atol=1e-10)


def test_energy_only_request(server, job):
    """A deriva=0 request still writes a well-formed zero gradient block."""
    srv, _port = server
    inp, out = job
    write_external_input(inp, make_water(), deriva=0)
    srv.compute(str(inp), str(out))
    energy_ha, grad = read_external_output(out)
    assert energy_ha != 0.0
    np.testing.assert_allclose(grad, 0.0)


# ═══════════════════════════════════════════════════════════════════════
#  Forwarding scripts on the Gaussian side
# ═══════════════════════════════════════════════════════════════════════

def run_relay(script, inp, out, port):
    env = dict(os.environ, MLIP_SERVER_PORT=str(port))
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / 'gau_scripts' / script),
         'layer', str(inp), str(out)],
        env=env, capture_output=True, text=True)


def test_generic_relay_forwards_request(server, job):
    """Gau_generic.py skips the model check and returns a result."""
    _srv, port = server
    inp, out = job
    write_external_input(inp, make_water())
    r = run_relay('Gau_generic.py', inp, out, port)
    assert r.returncode == 0, r.stderr
    energy_ha, grad = read_external_output(out)
    assert energy_ha != 0.0 and grad.shape == (3, 3)


def test_relay_rejects_model_mismatch(server, job):
    """A Gau_<model>.py wrapper refuses a model the server has not loaded."""
    _srv, port = server
    inp, out = job
    write_external_input(inp, make_water())
    r = run_relay('Gau_mace_off24.py', inp, out, port)
    assert r.returncode == 1, 'a model mismatch must fail loudly, never compute silently'
    assert 'mace_off24' in r.stderr and TEST_MODEL in r.stderr


# ═══════════════════════════════════════════════════════════════════════
#  Real-model reference case (skipped without its dependency or weights)
# ═══════════════════════════════════════════════════════════════════════

def test_dpa4_reference_script_runs():
    """test/dpa4.py is the reference for a real model and runs once weights exist."""
    pytest.importorskip('deepmd', reason='deepmd-kit is not installed; skipping the DPA4 case')
    if not os.environ.get('MLIP_MODEL_DPA4'):
        pytest.skip('MLIP_MODEL_DPA4 is not configured (config.env or environment)')
    r = subprocess.run([sys.executable, str(TEST_DIR / 'dpa4.py')],
                       cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
