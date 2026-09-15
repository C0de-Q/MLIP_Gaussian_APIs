"""test — test cases for adding an ASE model

Runnable tests for the workflow in the "Adding a model" section of the README
and in "Option A (declarative)" of docs/ADD_NEW_MODEL.md:

    custom_models.py       Plugin under test: registers emttest (the ASE EMT
                           calculator, no weights or downloads needed) and dpa4
                           (the real DeepMD registration)
    test_add_ase_model.py  pytest cases: plugin loading, server persistence,
                           energy and gradient output, and Gau_*.py forwarding
                           including the model name check
    dpa4.py                Reference script for the real DPA4 model

Run them with:

    pip install pytest
    pytest test/ -v

Only the dpa4.py cases need deepmd-kit and MLIP_MODEL_DPA4 weights, and they are
skipped automatically when either is missing. Everything else needs just numpy
and ase.
"""
