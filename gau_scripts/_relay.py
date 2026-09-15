#!/usr/bin/env python3
"""
gau_scripts/_relay.py — shared forwarding entry point for every Gau_*.py

Design constraint: RunMLIPgjf.sh must load the model into the persistent server
first. A Gau_*.py script only forwards Gaussian's request to that server and
never loads a model itself.

Responsibilities:
    1. Put the repository root on sys.path.
    2. Check that MLIP_SERVER_PORT is set and the server answers.
    3. Optionally check that the loaded model matches the script name.
    4. Forward the request and wait for the result.
"""

import os
import sys
import logging

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import log_utils  # importing it configures the root logger (stderr, kept by Gaussian)

logger = logging.getLogger('gau_relay')


def _require_server(method=None):
    """Return the server port, or exit with an error if it is unusable."""
    port = os.environ.get('MLIP_SERVER_PORT')
    if not port:
        logger.error(
            'MLIP_SERVER_PORT is not set. The model has to be loaded before '
            'Gaussian runs: start a server with RunMLIPgjf.sh first, for example '
            './RunMLIPgjf.sh job.gjf. Alternatively set MLIP_SERVER_PORT to point '
            'at an MLIP server that is already running.')
        raise SystemExit(1)

    from mlip_relay import server_method
    actual = server_method(int(port))
    if method and actual and actual != method:
        logger.error(
            'The server has "%s" loaded, but this script expects %s. Restart the '
            'server with RunMLIPgjf.sh -m %s, or point the Gaussian external at '
            'Gau_%s.py / Gau_generic.py instead.',
            actual, method, method, actual)
        raise SystemExit(1)
    return int(port)


def relay(method):
    """Forward with a model name check; used by Gau_<method>.py."""
    port = _require_server(method)
    _forward(port, method)


def relay_any():
    """Forward without checking the model name; used by Gau_generic.py."""
    port = _require_server()
    _forward(port, None)


def _forward(port, method=None):
    from gaussian_external import parse_gaussian_args
    from mlip_relay import client_mode

    filein, fileout, _msgfile, _fchkfile = parse_gaussian_args()
    try:
        client_mode(filein, fileout, port)
    except RuntimeError as e:
        logger.error('[%s] forwarding to the MLIP server failed: %s',
                     method or 'generic', e)
        sys.exit(1)
