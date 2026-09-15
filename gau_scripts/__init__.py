"""
gau_scripts — Gaussian External interface scripts (thin wrappers)

Each Gau_*.py is a one-line wrapper around gau_scripts/_relay.py that forwards
requests to the persistent server started by RunMLIPgjf.sh; Gaussian calls them
directly through the external keyword. Use Gau_generic.py when the job does not
care which model is loaded. See docs/ADD_NEW_MODEL.md for adding a model.
"""

__version__ = '0.2.0'
