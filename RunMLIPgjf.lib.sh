#!/bin/bash
# RunMLIPgjf.lib.sh — the implementation behind RunMLIPgjf.sh
#
# Sourced by RunMLIPgjf.sh, which stays short on purpose: everything that does
# work lives here, one function per step. Running this file directly does nothing.
#
#   load_config         read config.env and apply the built-in defaults
#   usage               print the help text
#   parse_args          command line options and the checks on them
#   collect_jobs        turn the target argument into the GJF_LIST array
#   export_server_env   environment handed to the MLIP server process
#   port_ready          is something listening on the server port?
#   cleanup_port        stop a leftover process holding that port
#   start_server        launch mlip_server.py and wait until it answers
#   stop_server         terminate the server (also called from the EXIT trap)
#   warmup_server       one throwaway calculation, so the model is loaded early
#   load_gaussian_env   source g16-env.sh, then make sure g16 is callable
#   run_jobs            run every .gjf from its own directory, logging each one
#   finish_jobs         print the summary and set the exit code
#
# Gaussian 16 itself comes from g16-env.sh next to RunMLIPgjf.sh (copy
# g16-env.example.sh): a module load, PATH, a site environment script, whatever
# your installation needs.
#
# The order in which these functions run is in RunMLIPgjf.sh, so the workflow
# stays readable without scrolling through the implementation.

# ── State shared by the functions below ──
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD=""
PORT=""
PYTHON_BIN=""
START_SERVER=1
KILL_PORT=1
WARMUP=1
LOG_DIR=""
JOB_LOG_DIR=""
TARGETS=()
GJF_LIST=()
SERVER_PID=""
SERVER_LOG=""
FAILED_JOBS=0
_SERVER_STARTED=0


usage() {
    cat <<'EOF'
Usage: ./RunMLIPgjf.sh [options] <job.gjf | directory>

Options:
  -m, --method NAME   model; defaults to METHOD from config.env, then mace_off24
  -p, --port PORT     server port (default 15556)
  -l, --log-dir DIR   write every log here; without it they go next to the .gjf
                      file, or into the directory that was passed in
  --python CMD        Python used to start the server (default $MLIP_PYTHON, then python3)
  --no-server         do not start a server; only run g16 against an existing one
  --no-kill           do not clean up a leftover process holding the port
  --no-warmup         skip the warm-up calculation (see below)
  --warmup            run it again, after --no-warmup
  -h, --help          show this help

The warm-up is one throwaway calculation right after the server is ready. Models
that load lazily (mace_polar, orbmol, ANI, AIMNet2, ...) pay for that load on
their first request, which would otherwise be the first step of your job; running
it here keeps the job's first step as fast as the rest and reports the load time.
EOF
}


# ── Configuration and command line ──

load_config() {
    if [ -f "$REPO_ROOT/config.env" ]; then
        # Disable set -u temporarily: the config file may reference undefined
        # variables, which must not abort the script.
        set +u
        set -a
        # shellcheck disable=SC1091
        . "$REPO_ROOT/config.env"
        set +a
        set -u
    fi
    # METHOD precedence: -m on the command line, then the environment or
    # config.env, then the built-in default.
    METHOD="${METHOD:-mace_off24}"
    PORT="${MLIP_SERVER_PORT:-15556}"
    PYTHON_BIN="${MLIP_PYTHON:-python3}"
}


parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -m|--method)
                METHOD="$2"
                shift 2
                ;;
            -p|--port)
                PORT="$2"
                shift 2
                ;;
            -l|--log-dir)
                LOG_DIR="$2"
                shift 2
                ;;
            --python)
                PYTHON_BIN="$2"
                shift 2
                ;;
            --no-server)
                START_SERVER=0
                shift
                ;;
            --no-kill)
                KILL_PORT=0
                shift
                ;;
            --no-warmup)
                WARMUP=0
                shift
                ;;
            --warmup)
                WARMUP=1
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                TARGETS+=("$1")
                shift
                ;;
        esac
    done

    if [ -n "$LOG_DIR" ]; then
        mkdir -p "$LOG_DIR" || { echo -e "\033[31mCannot create log directory: $LOG_DIR\033[0m"; exit 101; }
    fi

    if [ ${#TARGETS[@]} -eq 0 ]; then
        echo -e "\033[31mMissing argument: expected a .gjf file or a directory\033[0m"
        usage
        exit 101
    fi
    if [ ${#TARGETS[@]} -gt 1 ]; then
        echo -e "\033[31mOnly one argument is accepted: a .gjf file or a directory\033[0m"
        exit 101
    fi
}


collect_jobs() {
    local target="${TARGETS[0]}"
    if [ -f "$target" ]; then
        GJF_LIST=("$target")
        JOB_LOG_DIR="$(dirname "$target")"
    elif [ -d "$target" ]; then
        shopt -s nullglob
        GJF_LIST=("$target"/*.gjf)
        shopt -u nullglob
        if [ ${#GJF_LIST[@]} -eq 0 ]; then
            echo -e "\033[31mNo .gjf files in directory $target\033[0m"
            exit 101
        fi
        JOB_LOG_DIR="$target"
    else
        echo -e "\033[31mFile or directory does not exist: $target\033[0m"
        exit 101
    fi
    # --log-dir collects the server log and every job log in one place.
    if [ -n "$LOG_DIR" ]; then
        JOB_LOG_DIR="$LOG_DIR"
    fi
}


export_server_env() {
    export METHOD
    export MLIP_SERVER_PORT="$PORT"
    # Only forward MLIP_XYZ_OUT when it names a file: the server then writes no
    # structure file at all, and an empty value cannot trigger a failed write.
    if [ -n "${MLIP_XYZ_OUT:-}" ]; then
        export MLIP_XYZ_OUT
    else
        unset MLIP_XYZ_OUT
    fi
}


# ── MLIP server ──

port_ready() {
    python3 -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('127.0.0.1', $PORT)); s.close()" \
        >/dev/null 2>&1
}


# Clean up a leftover process holding the port (on by default; --no-kill).
cleanup_port() {
    if [ "$KILL_PORT" -eq 0 ]; then
        echo -e "\033[31m[RunMLIPgjf] Port $PORT is already in use and --no-kill was given; exiting.\033[0m"
        exit 101
    fi
    if ! port_ready; then
        return 0
    fi

    echo "[RunMLIPgjf] Port $PORT is in use; stopping the process holding it..."
    local pid=""
    pid=$(python3 -c "
    import os, signal, glob

    def find_pids_on_port(port):
        port_hex = '%04X' % port
        inodes = set()
        for path in ('/proc/net/tcp', '/proc/net/tcp6'):
            try:
                with open(path) as f:
                    next(f)
                    for line in f:
                        parts = line.split()
                        if len(parts) > 9 and parts[1].endswith(':' + port_hex):
                            inodes.add(parts[9])
            except OSError:
                pass
        pids = set()
        for p in glob.glob('/proc/[0-9]*'):
            try:
                for fd in glob.glob(p + '/fd/*'):
                    try:
                        link = os.readlink(fd)
                    except OSError:
                        continue
                    if link.startswith('socket:[') and link[8:-1] in inodes:
                        pids.add(int(p.rsplit('/', 1)[1]))
            except OSError:
                continue
        return sorted(pids)

    for p in find_pids_on_port($PORT):
        try:
            os.kill(p, signal.SIGTERM)
            print(p)
        except OSError:
            pass
    " 2>/dev/null || true)

    if [ -n "$pid" ]; then
        echo "[RunMLIPgjf] Sent SIGTERM to PID: $pid"
    else
        # Fallback tools, which may not be installed: lsof / fuser
        lsof -ti tcp:"$PORT" 2>/dev/null | xargs -r kill -TERM 2>/dev/null || true
        fuser -k -n tcp "$PORT" 2>/dev/null || true
    fi

    # Wait up to 10s for the port to be released, then force it.
    local waited=0
    while [ $waited -lt 20 ] && port_ready; do
        sleep 0.5
        waited=$((waited + 1))
    done
    if port_ready; then
        echo "[RunMLIPgjf] Force-terminated the process holding the port"
        lsof -ti tcp:"$PORT" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
        fuser -k -9 -n tcp "$PORT" 2>/dev/null || true
        sleep 1
    else
        echo "[RunMLIPgjf] Port $PORT is free."
    fi
}


start_server() {
    if [ "$START_SERVER" -eq 0 ]; then
        if ! port_ready; then
            echo -e "\033[31m[RunMLIPgjf] --no-server was given, but nothing is listening on port $PORT\033[0m"
            exit 101
        fi
        echo "[RunMLIPgjf] Reusing the running server (METHOD=$METHOD, PORT=$PORT)"
        return 0
    fi

    cleanup_port
    if port_ready; then
        echo -e "\033[31m[RunMLIPgjf] Port $PORT is still in use; cannot start the server."
        echo "  Free the port manually and try again.\033[0m"
        exit 101
    fi

    SERVER_LOG="$JOB_LOG_DIR/mlip_server_$$.log"
    echo "[RunMLIPgjf] Starting the MLIP server: METHOD=$METHOD, PORT=$PORT"
    echo "[RunMLIPgjf] Server log: $SERVER_LOG"

    local run_cmd="$PYTHON_BIN $REPO_ROOT/mlip_server.py"
    if [ -n "${MLIP_CONDA_SETUP:-}" ]; then
        run_cmd="${MLIP_CONDA_SETUP} && ${run_cmd}"
    fi
    bash -c "$run_cmd" > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    _SERVER_STARTED=1
    echo "[RunMLIPgjf] Server PID: $SERVER_PID"

    # Wait for readiness: up to 240 × 0.5s = 120s.
    local waited=0
    while [ $waited -lt 240 ]; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo -e "\033[31m[RunMLIPgjf] The server failed to start; log follows:\033[0m"
            cat "$SERVER_LOG" 2>/dev/null
            exit 1
        fi
        if port_ready; then
            echo "[RunMLIPgjf] Server is ready."
            return 0
        fi
        sleep 0.5
        waited=$((waited + 1))
    done

    echo -e "\033[31m[RunMLIPgjf] Timed out waiting for the server (120s); log follows:\033[0m"
    cat "$SERVER_LOG" 2>/dev/null
    stop_server
    exit 1
}


stop_server() {
    if [ "$_SERVER_STARTED" -eq 0 ]; then
        return 0
    fi
    _SERVER_STARTED=0        # idempotent: the EXIT trap may call this a second time
    echo "[RunMLIPgjf] Stopping the MLIP server (PID: $SERVER_PID)..."
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    local waited=0
    while [ $waited -lt 20 ]; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "[RunMLIPgjf] The server has exited."
            return 0
        fi
        sleep 0.5
        waited=$((waited + 1))
    done
    kill -9 "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    echo "[RunMLIPgjf] The server had to be killed."
}


# ── Warm-up ──

# One throwaway request, so a lazily loaded model is in memory before the first
# job. Uses the same relay the Gaussian jobs use, which also checks that the
# forwarding path works at all.
warmup_server() {
    if [ "$WARMUP" -eq 0 ]; then
        echo "[RunMLIPgjf] Warm-up skipped (--no-warmup)."
        return 0
    fi

    local dir infile outfile started_ms elapsed_ms
    dir="$(mktemp -d "${TMPDIR:-/tmp}/mlip_warmup_XXXXXX")"
    infile="$dir/water.dat"
    outfile="$dir/water.out"
    # A water molecule in Bohr, in the format Gaussian writes: deriva=1 asks for
    # the gradient too, so both code paths are exercised.
    cat > "$infile" <<'EOF'
3 1 0 1
8    0.000000000000    0.000000000000    0.000000000000    0.000000
1    0.000000000000    0.000000000000    1.810000000000    0.000000
1    0.000000000000    1.750000000000   -0.450000000000    0.000000
EOF

    echo "[RunMLIPgjf] Warm-up: loading the model with one throwaway calculation..."
    started_ms=$(date +%s%3N)
    if ( export MLIP_SERVER_PORT="$PORT" MLIP_WARMUP=1
         $PYTHON_BIN "$REPO_ROOT/gau_scripts/Gau_generic.py" R "$infile" "$outfile"
       ) > "$dir/relay.log" 2>&1; then
        elapsed_ms=$(( $(date +%s%3N) - started_ms ))
        echo "[RunMLIPgjf] Warm-up finished in ${elapsed_ms} ms; the model stays in memory."
        if [ -n "$SERVER_LOG" ] && [ -f "$SERVER_LOG" ]; then
            grep -E "Warm-up|Model loaded|ASE calculator loaded" "$SERVER_LOG" \
                | tail -2 | sed 's/^/[RunMLIPgjf]   /'
        fi
    else
        echo -e "\033[31m[RunMLIPgjf] Warm-up failed; the model cannot be evaluated.\033[0m"
        sed 's/^/[RunMLIPgjf]   /' "$dir/relay.log" 2>/dev/null
        if [ -n "$SERVER_LOG" ] && [ -f "$SERVER_LOG" ]; then
            tail -5 "$SERVER_LOG" | sed 's/^/[RunMLIPgjf]   /'
        fi
        echo "  Fix the model, or run the jobs anyway with --no-warmup."
        rm -rf "$dir"
        exit 1
    fi
    rm -rf "$dir"
}


# ── Gaussian side ──

# Source g16-env.sh, then fail early when g16 is still not callable.
load_gaussian_env() {
    G16_ENV_FILE="${G16_ENV_FILE:-$REPO_ROOT/g16-env.sh}"
    if [ -f "$G16_ENV_FILE" ]; then
        echo "[RunMLIPgjf] Loading the Gaussian environment from $G16_ENV_FILE"
        # Site environment scripts reference variables such as PERLLIB that are
        # often unset; disable set -u while sourcing so that cannot abort the run.
        set +u
        # shellcheck disable=SC1090
        source "$G16_ENV_FILE"
        set -u
    else
        echo "[RunMLIPgjf] No $G16_ENV_FILE; assuming g16 is already on PATH."
    fi

    if ! command -v g16 >/dev/null 2>&1; then
        echo -e "\033[31m[RunMLIPgjf] g16 is not available on PATH.\033[0m"
        echo "  Either put g16 on PATH before running this script, or create"
        echo "  $REPO_ROOT/g16-env.sh (copy g16-env.example.sh) with the environment"
        echo "  your Gaussian installation needs."
        exit 101
    fi
}


run_jobs() {
    FAILED_JOBS=0
    local gjf gjf_abs job_dir job_name job_log log_dir_abs scratch rc
    log_dir_abs="$(cd "$JOB_LOG_DIR" && pwd)"
    for gjf in "${GJF_LIST[@]}"; do
        # g16 runs inside the directory that holds the .gjf, so Gaussian's own
        # files (%chk and anything else it writes) land next to the job instead of
        # in the caller's working directory. Paths are made absolute first, so the
        # log can still live elsewhere (--log-dir).
        gjf_abs="$(cd "$(dirname "$gjf")" && pwd)/$(basename "$gjf")"
        job_dir="$(dirname "$gjf_abs")"
        job_name=$(basename "$gjf_abs" .gjf)
        job_log="$log_dir_abs/$job_name.log"
        scratch="/tmp/g16_${job_name}_$$"
        mkdir -p "$scratch"
        export GAUSS_SCRDIR="$scratch"

        echo "========================================"
        echo "[RunMLIPgjf] Starting job: $gjf  $(date)"
        echo "GAUSS_SCRDIR=$GAUSS_SCRDIR" > "$job_log"

        if ( cd "$job_dir" && g16 < "$gjf_abs" > "$job_log" 2>&1 ); then
            rc=0
        else
            rc=$?
        fi
        date >> "$job_log"

        if [ "$rc" -eq 0 ]; then
            echo "[RunMLIPgjf] Job finished: $gjf -> $job_log  $(date)"
        else
            echo -e "\033[31m[RunMLIPgjf] Job failed (rc=$rc): $gjf\033[0m"
            FAILED_JOBS=$((FAILED_JOBS + 1))
        fi
        rm -rf "$scratch"
    done
}


# ── Wrap-up ──

finish_jobs() {
    echo "========================================"
    if [ "$FAILED_JOBS" -eq 0 ]; then
        echo "[RunMLIPgjf] All ${#GJF_LIST[@]} job(s) finished."
        exit 0
    else
        echo "[RunMLIPgjf] Finished with $FAILED_JOBS out of ${#GJF_LIST[@]} job(s) failed."
        exit 1
    fi
}


if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "RunMLIPgjf.lib.sh is a function library; run ./RunMLIPgjf.sh instead." >&2
    exit 1
fi
