#!/bin/bash
# g16-env.sh — Gaussian 16 environment for RunMLIPgjf.sh
#
# This is the half of the launcher that belongs to you. RunMLIPgjf.sh sources the
# file next to it (g16-env.sh) right before the first g16 call, so whatever you
# set here is visible to g16 and to nothing else: the MLIP server was started
# earlier with its own Python environment, which keeps Gaussian libraries from
# interfering with it.
#
# Usage:
#     cp g16-env.example.sh g16-env.sh
#     # edit g16-env.sh for your site
#     ./RunMLIPgjf.sh job.gjf
#
# g16-env.sh is git-ignored, so your local paths stay out of the repository. When
# it does not exist, RunMLIPgjf.sh expects g16 to be on PATH already and stops
# with a clear message if it is not.
#
# Uncomment and adapt whichever block matches your installation.

# --- Option 1: module system ---
# module load gaussian/16

# --- Option 2: your site ships an environment script ---
# source /share/apps/gaussian/g16-env.sh
# (wrap it in [ -f ... ] if the file may be missing on some machines)

# --- Option 3: set the paths by hand ---
# export g16root=/share/apps/gaussian
# export GAUSS_EXEDIR=$g16root/g16
# export GAUSS_LEXEDIR=$g16root/g16/bsd
# export PATH=$GAUSS_EXEDIR:$PATH
# export LD_LIBRARY_PATH=$GAUSS_EXEDIR:$LD_LIBRARY_PATH

# --- Option 4: a wrapper or a differently named binary ---
# g16() { exec /share/apps/gaussian/g16/g16 "$@"; }

# Gaussian scratch space is created per job by RunMLIPgjf.sh
# (GAUSS_SCRDIR=/tmp/g16_<JobName>_<pid>), so normally leave it alone. Uncomment
# the next line if your site's jobs need a larger stack:
# ulimit -s unlimited

# Notes:
# * The file is sourced with `set -u` disabled, so referring to variables that may
#   be unset (PERLLIB and friends inside g16.profile) is safe.
# * It is sourced, not executed, so use bash syntax and do not add exit calls.
