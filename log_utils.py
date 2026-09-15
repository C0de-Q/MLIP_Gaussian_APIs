#!/usr/bin/env python3
"""
log_utils.py — shared logging configuration

Every module logs through logging.getLogger(__name__) with the format
    2026-08-24 21:30:00 INFO    [mlip_server] Step natoms=2 E=3.26 eV

The level comes from the MLIP_LOG_LEVEL environment variable (DEBUG/INFO/
WARNING/ERROR, default INFO). Importing this module configures the root logger;
it is idempotent and never adds a second handler.
"""

import logging
import os
import sys

_LOG_FORMAT = '%(asctime)s %(levelname)-7s [%(name)s] %(message)s'
_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


def setup_logging(level=None, stream=None):
    """Configure the root logger; repeated calls do not add extra handlers.

    Args:
        level:  log level as a string or int; defaults to MLIP_LOG_LEVEL.
        stream: output stream; defaults to sys.stderr, which Gaussian captures
                into the job log.
    """
    if level is None:
        level = os.environ.get('MLIP_LOG_LEVEL', 'INFO')
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        handler = logging.StreamHandler(stream or sys.stderr)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        root.addHandler(handler)
    return root


def get_logger(name):
    """Return a named logger that uses the shared configuration."""
    return logging.getLogger(name)


setup_logging()
