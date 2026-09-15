#!/bin/bash
# RunMLIPgjf.sh — start a persistent MLIP server and run Gaussian 16 jobs
#
#   ./RunMLIPgjf.sh job.gjf          run one job
#   ./RunMLIPgjf.sh /path/to/dir     run every *.gjf in a directory, serially
#   ./RunMLIPgjf.sh --help           list the options
#
# The block at the bottom is the whole workflow, in order. The functions it calls,
# and the options each of them supports, live in RunMLIPgjf.lib.sh next to this
# file; the Gaussian 16 environment lives in g16-env.sh (copy g16-env.example.sh
# and edit that). See README.md for the details.
#
# Exit code 0 when every job succeeded, 1 otherwise.

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# shellcheck source=RunMLIPgjf.lib.sh
source "$SCRIPT_DIR/RunMLIPgjf.lib.sh"


# ── Workflow ──

load_config                 # config.env plus the built-in defaults
parse_args "$@"             # options, and the checks on them
collect_jobs                # the target argument becomes GJF_LIST
export_server_env           # environment handed to the server process

trap 'stop_server' EXIT     # always shut the server down on the way out

start_server                # mlip_server.py, waiting until it answers
warmup_server               # one throwaway call: load the model before the jobs
load_gaussian_env           # g16-env.sh first, then check that g16 is callable
run_jobs                    # g16 for every .gjf, logging each one

finish_jobs                 # summary and exit code
