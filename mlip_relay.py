#!/usr/bin/env python3
"""
mlip_relay.py — TCP relay client for the Gaussian External interface

Every gau_scripts/Gau_*.py script forwards its request to the persistent MLIP
server started by RunMLIPgjf.sh, which loads the model once. MLIP_SERVER_PORT
must be set; when no server answers, the relay reports an error instead of
silently computing locally, because the model has to be loaded up front.
"""

import os
import sys


def server_method(port):
    """Probe which model the server has loaded (sends PING, computes nothing)."""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10)
    try:
        sock.connect(('127.0.0.1', port))
        sock.sendall(b'PING\n')
        response = b''
        while b'\n' not in response:
            chunk = sock.recv(4096)
            if not chunk:
                break
            response += chunk
    finally:
        sock.close()
    parts = response.decode().strip().split()
    if parts and parts[0] == 'OK':
        return parts[1] if len(parts) > 1 else None
    raise RuntimeError(f'MLIP server probe failed: {response!r}')


def client_mode(filein, fileout, port, warmup=False):
    """Forward one compute request to the server, returning its loaded model.

    With warmup=True the server labels the line it logs as a Warm-up instead of a
    step, so that one throwaway calculation does not look like a job step. Setting
    MLIP_WARMUP=1 in the environment has the same effect, which is how the shell
    runner marks its warm-up call.
    """
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(300)
    sock.connect(('127.0.0.1', port))
    if warmup or os.environ.get('MLIP_WARMUP') == '1':
        request = 'WARMUP ' + filein + '\n' + fileout + '\n'
    else:
        request = filein + '\n' + fileout + '\n'
    sock.sendall(request.encode())
    response = b''
    while b'\n' not in response:
        chunk = sock.recv(4096)
        if not chunk:
            raise RuntimeError('Server closed connection')
        response += chunk
    sock.close()
    text = response.decode().strip()
    parts = text.split()
    if not parts or parts[0] != 'OK':
        raise RuntimeError('Server error: %s' % text)
    return parts[1] if len(parts) > 1 else None


if __name__ == '__main__':
    from gaussian_external import parse_gaussian_args

    filein, fileout, _, _ = parse_gaussian_args()
    port = os.environ.get('MLIP_SERVER_PORT')
    if not port:
        raise SystemExit(
            '[mlip_relay] MLIP_SERVER_PORT is not set. Start a server first with '
            'RunMLIPgjf.sh; the model must be loaded before Gaussian runs.')
    client_mode(filein, fileout, int(port))
