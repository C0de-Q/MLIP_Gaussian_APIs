#!/usr/bin/env python3
"""
mlip_relay.py — TCP relay client for Gaussian External interface

作为 Gau_*.py 被 Gaussian 调用时，将计算请求转发给持久化 GPU server。
需设置环境变量 MLIP_SERVER_PORT。

Gaussian 调用方式:
    mlip_relay.py  layer  inputfile  outputfile

如果未设置 MLIP_SERVER_PORT，则 fallback 到本地 CPU 计算（调用 calculators.py）。
"""

import sys
import os


def client_mode(filein, fileout, port):
    """将计算请求通过 TCP 转发给持久化 server。"""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(300)
    sock.connect(('127.0.0.1', port))
    sock.sendall((filein + '\n' + fileout + '\n').encode())
    response = b''
    while b'\n' not in response:
        chunk = sock.recv(4096)
        if not chunk:
            raise RuntimeError('Server closed connection')
        response += chunk
    sock.close()
    if not response.startswith(b'OK'):
        raise RuntimeError('Server error: %s' % response.decode().strip())


def cpu_fallback(filein, fileout, method=None):
    """CPU 本地计算 fallback。按 METHOD 环境变量或参数选择计算器。"""
    if method is None:
        method = os.environ.get('METHOD', 'mace_off24')

    sys.path.insert(0, '/share/home/CodeQ/Benchmark')
    from bin.gaussian_external import get_external_coord, write_xyz, write_external_output
    from bin.calculators import METHODS
    from bin.constants import EV_HA, FORCE_UNIT_CONST
    from ase.io import read

    if method not in METHODS:
        raise ValueError(f"Unknown fallback method '{method}'. "
                         f"Available: {sorted(METHODS.keys())}")

    compute_fn = METHODS[method]
    ele, coordlist, atom_charges, deriva, charge, spin = get_external_coord(filein)
    write_xyz(ele, coordlist, charge, spin, 'gau_mlatom.xyz')
    atoms = read('gau_mlatom.xyz')
    ene_ev = compute_fn(atoms, charge, spin)

    if deriva == 0:
        write_external_output(fileout, ene_ev / EV_HA, None, len(ele), deriva)
    elif deriva == 1:
        forces = atoms.get_forces()
        grad = [[-f[0] / FORCE_UNIT_CONST, -f[1] / FORCE_UNIT_CONST, -f[2] / FORCE_UNIT_CONST] for f in forces]
        write_external_output(fileout, ene_ev / EV_HA, grad, len(ele), deriva)


if __name__ == '__main__':
    if len(sys.argv) >= 4:
        filein = sys.argv[2]
        fileout = sys.argv[3]
    elif len(sys.argv) == 3:
        filein = sys.argv[1]
        fileout = sys.argv[2]
    else:
        print('Usage: mlip_relay.py [layer] inputfile outputfile', file=sys.stderr)
        sys.exit(1)

    port = os.environ.get('MLIP_SERVER_PORT')
    if port:
        client_mode(filein, fileout, int(port))
    else:
        print('[mlip_relay] No MLIP_SERVER_PORT set, using CPU fallback', file=sys.stderr)
        cpu_fallback(filein, fileout)
