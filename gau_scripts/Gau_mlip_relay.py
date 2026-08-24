#!/usr/bin/env python3
"""Gau_mlip_relay.py — TCP relay to persistent GPU server (for Gaussian External)"""
import sys
import os
sys.path.insert(0, '/share/home/CodeQ/Benchmark')
from bin.mlip_relay import client_mode

if __name__ == '__main__':
    filein, fileout = sys.argv[2], sys.argv[3]
    port = os.environ.get('MLIP_SERVER_PORT')
    if port:
        client_mode(filein, fileout, int(port))
    else:
        # 无 GPU server → CPU fallback
        print('[Gau_mlip_relay] MLIP_SERVER_PORT not set, using CPU fallback', file=sys.stderr)
        from bin.mlip_relay import cpu_fallback
        cpu_fallback(filein, fileout)
