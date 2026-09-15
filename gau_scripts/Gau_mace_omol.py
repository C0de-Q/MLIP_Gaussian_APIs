#!/usr/bin/env python3
"""Gaussian External interface for MACE-OMol extra-large (start the MLIP server first)."""
from _relay import relay

if __name__ == '__main__':
    relay('mace_omol')
