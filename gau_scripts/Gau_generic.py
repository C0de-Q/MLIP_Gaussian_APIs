#!/usr/bin/env python3
"""Generic Gaussian External interface: the server decides which model runs.

Gaussian route: external="python /path/to/gau_scripts/Gau_generic.py"
Whatever model the server loaded is the one used, as chosen by RunMLIPgjf.sh
(METHOD) or -m. The model name is not checked here.
"""
from _relay import relay_any

if __name__ == '__main__':
    relay_any()
