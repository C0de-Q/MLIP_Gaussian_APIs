#!/usr/bin/env python3
"""
mlip_server.py — persistent MLIP server for Gaussiancal culations

Started by RunMLIPgjf.sh, it accepts inference requests over TCP. The model is
selected with the METHOD environment variable (or -m for RunMLIPgjf.sh) and must
be registered in calculators.METHODS.

MLIPServer contains no model-specific logic; the compute path is decided by the
registration in the calculators package:
  - Persistent ASE path: declarative ASE models (ASE_MODEL_SPECS, such as the
    MACE and DPA families) build their calculator once at startup and reuse it
    for every request.
  - Generic path: functional models (orbmol_v2, orbmol, ANI, AIMNet2, ...) call
    calculators.compute / compute_gradient per request. Python models that can be
    cached are loaded once through the in-process cache; CLI models (aiqm3,
    d4ani, gxtb) run an external program per step and cannot stay resident.

Environment variables:
    MLIP_SERVER_PORT   listen port (default 15556)
    MLIP_SERVER_HOST   listen address (default 127.0.0.1)
    MLIP_SERVER_DEVICE device for model evaluation (default: cuda when a GPU is
                       available, otherwise cpu)
    METHOD             model selection
    MLIP_XYZ_OUT       write the final structure to this XYZ file on shutdown
    MLIP_LOG_LEVEL     log level (DEBUG/INFO/WARNING/ERROR, default INFO)

Model weight paths come from constants.py / config.env (see config.example.env
in the repository root).
"""

import socket
import sys
import os
import signal
import threading
import time
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from ase import Atoms
from ase.io import write as ase_write

try:  # torch is only needed for the persistent ASE path and CUDA cache release
    import torch
except ImportError:  # pragma: no cover
    torch = None

from constants import ELEMENTS, server_device
from calculators import (get_ase_spec, make_ase_calculator, prepare_ase_atoms,
                         take_load_time)
from gaussian_external import get_external_coord, unique_scratch_base, write_external_output_ev
import log_utils  # importing it configures the root logger

logger = logging.getLogger('mlip_server')


def _empty_cuda_cache():
    """Release the CUDA caching allocator's memory when CUDA is in use."""
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _generic_compute(method, atoms, charge, spin, deriva):
    """Generic compute path: call the unified entry points in calculators.

    Returns:
        energy_ev: float
        forces_ev_ang: np.ndarray (natoms,3) | None
    """
    from calculators import compute as ml_compute, compute_gradient

    base = unique_scratch_base(method)
    energy_ev = ml_compute(method, atoms, charge, spin, base=base)
    forces = None
    if deriva == 1:
        grad = compute_gradient(method, atoms, charge, spin, base=base)
        if grad is not None:
            # calculators returns dE/dr in eV/Å, so negate it to get forces in
            # eV/Å; write_external_output_ev converts to Ha/Bohr for Gaussian.
            grad = np.asarray(grad, dtype=np.float64)
            # A wrong shape used to fall through and silently produce zero forces,
            # which looks to Gaussian like a converged structure. Fail instead.
            if grad.shape != (len(atoms), 3):
                raise ValueError(
                    f'{method} returned a gradient with shape {grad.shape}, '
                    f'expected {(len(atoms), 3)}')
            forces = -grad
    return energy_ev, forces


# ═══════════════════════════════════════════════════════════════════════
#  MLIP Server
# ═══════════════════════════════════════════════════════════════════════

class MLIPServer:
    """Persistent MLIP inference server (no model-specific logic)."""

    def __init__(self, method, host='127.0.0.1', port=15556,
                 xyz_out=None):
        self.method = method
        self.host = host
        self.port = port
        self.xyz_out = xyz_out
        self.model_lock = threading.Lock()
        self.ase_calc = None
        self._ase_spec = None
        self.device = None
        # Last computed state, used to write the final structure on shutdown.
        self.last_atoms = None
        self.last_energy = None
        self.last_forces = None
        self._shutdown_flag = threading.Event()
        self._load_ase_persistent()

    def _load_ase_persistent(self):
        """Declarative ASE models: build the calculator once, then reuse it.

        Whether this path applies is decided by calculators.ASE_MODEL_SPECS;
        this class never inspects method names.
        """
        spec = get_ase_spec(self.method)
        if spec is None:
            return
        self._ase_spec = spec
        self.device = server_device()
        logger.info('Loading the ASE calculator for %s '
                    '(once; reused by every later request, device=%s)...',
                    self.method, self.device)
        t_load = time.perf_counter()
        if spec.get('device'):
            # Only models whose spec declares a device key receive this argument.
            self.ase_calc = make_ase_calculator(spec, device=self.device)
        else:
            self.ase_calc = make_ase_calculator(spec)
        logger.info('ASE calculator loaded in %.2fs',
                    time.perf_counter() - t_load)

    def compute(self, filein, fileout, warmup=False):
        """Handle a single Gaussian External request.

        The reported time is the computation only: time spent loading a model
        inside this request (lazily loaded models do that on their first call) is
        split out and printed as `load=`. A warm-up request is labelled as such so
        it does not look like the first step of a job.
        """
        t_start = time.time()
        take_load_time()      # drop anything left over from an earlier request
        eles, coords, _atom_charges, deriva, charge, spin = get_external_coord(filein)

        # Build the ASE Atoms straight from the arrays (no intermediate XYZ file).
        symbols = [ELEMENTS[int(z)] for z in eles]
        atoms = Atoms(symbols=symbols, positions=coords)

        if self.ase_calc is not None:
            # ── Persistent ASE path (calculator already loaded, reused) ──
            prepare_ase_atoms(atoms, charge, spin, self._ase_spec)
            # Deliberately not wrapped in torch.no_grad(): ASE calculators such
            # as MACE rely on autograd when computing forces, and no_grad would
            # fail with "element 0 of tensors does not require grad".
            with self.model_lock:
                atoms.calc = self.ase_calc
                energy = atoms.get_potential_energy()
                forces = None if deriva == 0 else atoms.get_forces()
        else:
            # ── Generic path (functional model, called per request) ──
            energy, forces = _generic_compute(
                self.method, atoms, charge, spin, deriva)

        elapsed = time.time() - t_start
        load_seconds = take_load_time()
        logger.info('%-7s natoms=%d  E=%.8f eV  grad=%d  time=%.3fs%s',
                    'Warm-up' if warmup else 'Step', len(eles), energy, deriva,
                    elapsed - load_seconds,
                    f'  load={load_seconds:.2f}s' if load_seconds > 0.005 else '')

        write_external_output_ev(fileout, energy, forces, deriva, len(eles))

        # Remember the last state.
        if self.last_atoms is not None:
            del self.last_atoms
        self.last_atoms = atoms.copy()
        self.last_energy = energy
        self.last_forces = forces

        # Release GPU memory.
        del atoms, forces
        _empty_cuda_cache()

    def handle_client(self, conn):
        """Handle one TCP client connection."""
        try:
            data = b''
            while b'\n' not in data:
                chunk = conn.recv(4096)
                if not chunk:
                    return
                data += chunk
            first_line = data.split(b'\n', 1)[0].decode().strip()

            if first_line == 'PING':
                # Lightweight probe: report the model loaded in the server.
                conn.sendall(f'OK {self.method}\n'.encode())
            else:
                warmup = first_line.startswith('WARMUP ')
                if warmup:
                    first_line = first_line[len('WARMUP '):].strip()
                filein = first_line
                data = data.split(b'\n', 1)[1]
                while b'\n' not in data:
                    chunk = conn.recv(4096)
                    if not chunk:
                        return
                    data += chunk
                fileout = data.split(b'\n', 1)[0].decode().strip()

                self.compute(filein, fileout, warmup=warmup)
                conn.sendall(f'OK {self.method}\n'.encode())
        except Exception as e:
            logger.exception('Request failed: %s', e)
            try:
                conn.sendall(('ERROR: %s\n' % str(e)).encode())
            except Exception:
                pass
        finally:
            conn.close()

    def _save_final_xyz(self):
        """Write the last structure to the XYZ file, if one was requested.

        Nothing is written unless MLIP_XYZ_OUT names a file, so a normal run
        leaves no .xyz file behind.
        """
        if self.last_atoms is None or not self.xyz_out:
            return
        try:
            info = self.last_atoms.info
            info['energy_eV'] = self.last_energy
            if self.last_forces is not None:
                self.last_atoms.arrays['forces'] = self.last_forces
            ase_write(self.xyz_out, self.last_atoms)
            logger.info('Final structure saved to %s', self.xyz_out)
        except Exception as e:
            logger.warning('Failed to save XYZ: %s', e)

    def run(self):
        """Run the server accept loop."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(8)
        self.sock.settimeout(1.0)
        logger.info('Listening on %s:%s', self.host, self.port)

        with ThreadPoolExecutor(max_workers=4) as executor:
            while not self._shutdown_flag.is_set():
                try:
                    conn, addr = self.sock.accept()
                    executor.submit(self.handle_client, conn)
                except socket.timeout:
                    continue
                except Exception as e:
                    if not self._shutdown_flag.is_set():
                        logger.warning('Accept error: %s', e)

            executor.shutdown(wait=True)

        self._save_final_xyz()
        self.sock.close()
        logger.info('Shut down.')

    def shutdown(self):
        """Ask the accept loop to stop (used by the signal handlers)."""
        self._shutdown_flag.set()


# ═══════════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════════

_server_ref = None


def main():
    global _server_ref
    method = os.environ.get('METHOD', 'mace_off24')
    port = int(os.environ.get('MLIP_SERVER_PORT', 15556))
    host = os.environ.get('MLIP_SERVER_HOST', '127.0.0.1')
    # An empty MLIP_XYZ_OUT means "do not write a structure file".
    xyz_out = os.environ.get('MLIP_XYZ_OUT') or None

    from calculators import METHODS as CALCULATOR_METHODS
    available = sorted(CALCULATOR_METHODS)
    if method not in available:
        logger.error('Unknown METHOD=%s. Available: %s', method, available)
        sys.exit(1)

    if get_ase_spec(method):
        logger.info('%s uses the persistent ASE path.', method)
    else:
        logger.info('%s uses the generic path '
                    '(functional model, cached inside compute when possible).',
                    method)

    server = MLIPServer(method=method, host=host, port=port, xyz_out=xyz_out)
    _server_ref = server

    def _handle_signal(signum, frame):
        logger.info('Received shutdown signal.')
        server.shutdown()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    server.run()


if __name__ == '__main__':
    main()
