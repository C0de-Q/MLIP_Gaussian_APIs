#!/usr/bin/env python3
"""Gaussian External interface for ORB-Mol v2 (start the MLIP server first; persistent GPU path)."""
from _relay import relay

if __name__ == '__main__':
    relay('orbmol_v2')
