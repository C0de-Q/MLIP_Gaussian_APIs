#!/bin/bash
# RunG16GPU_orb.sh — 本地运行 Gaussian16 + orbmol_v2 MLIP GPU server (无需 PBS 调度器)
#
# 用法: ./RunG16GPU_orb.sh [工作目录]
#   默认处理当前目录下所有 *.gjf，串行逐个运行（一个 g16 结束再跑下一个）
#
# 工作流:
#   1. 清理端口残留
#   2. 通过 Apptainer 启动 mlip_server.py（orbmol_v2 GPU server，带 --nv）
#   3. 等待 server 就绪
#   4. 依次运行目录内每个 .gjf（g16 < gjf > JobName.log，等待其结束）
#   5. 自动清理 server

set -e

WORKDIR="${1:-.}"
cd "$WORKDIR" || { echo -e "\033[31m无法进入目录: $WORKDIR\033[0m"; exit 101; }

GJF_LIST=(*.gjf)
if [[ ! -e "${GJF_LIST[0]}" ]]; then
    echo -e "\033[31m目录 $WORKDIR 中没有 .gjf 文件\033[0m"
    exit 101
fi

export METHOD="orbmol_v2"
export MLIP_SERVER_PORT=15556
MLIP_HOST="127.0.0.1"
SIF_IMAGE="/share/home/CodeQ/docker/orbmol"
SERVER_LOG="$(pwd)/mlip_server_$$.log"

echo "[RunG16GPU_orb] 目录 $WORKDIR 下共 ${#GJF_LIST[@]} 个 .gjf 任务"

# ── 清理端口残留（失败不致命，set -e 下不能因此退出）──
_cleanup_port() {
    local pid
    pid=$(python -c "
    import socket, os, signal
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(('$MLIP_HOST', $MLIP_SERVER_PORT))
        s.close()
    except OSError:
        import subprocess, re
        try:
            out = subprocess.check_output(['ss', '-tlnp'], text=True)
        except Exception:
            out = ''
        for line in out.splitlines():
            if ':$MLIP_SERVER_PORT' in line:
                m = re.search(r'pid=(\d+)', line)
                if m:
                    p = int(m.group(1))
                    os.kill(p, signal.SIGTERM)
                    print(p)
                    break
    " 2>/dev/null || true)
    if [ -n "$pid" ]; then
        echo "[RunG16GPU_orb] Killed leftover process on port $MLIP_SERVER_PORT (PID: $pid)"
        sleep 1
        return 0
    fi
    # ss 缺失/无权限时用 fuser 兜底清理
    if command -v fuser >/dev/null 2>&1; then
        fuser -k -n tcp "$MLIP_SERVER_PORT" 2>/dev/null && \
            echo "[RunG16GPU_orb] Killed leftover process on port $MLIP_SERVER_PORT (fuser)" && sleep 1
    fi
}

# ── 清理 MLIP server ──
_MLIP_SERVER_PID=""

stop_mlip_server() {
    if [ -z "$_MLIP_SERVER_PID" ]; then
        return 0
    fi

    echo "[RunG16GPU_orb] Stopping MLIP server (PID: $_MLIP_SERVER_PID)..."

    # SIGTERM 触发优雅关闭
    kill -TERM $_MLIP_SERVER_PID 2>/dev/null || true

    # 等待最多 10s
    local waited=0
    while [ $waited -lt 20 ]; do
        if ! kill -0 $_MLIP_SERVER_PID 2>/dev/null; then
            echo "[RunG16GPU_orb] Server exited gracefully."
            return 0
        fi
        sleep 0.5
        waited=$((waited + 1))
    done

    # 强制 kill
    if kill -0 $_MLIP_SERVER_PID 2>/dev/null; then
        echo "[RunG16GPU_orb] Force killing server..."
        kill -9 $_MLIP_SERVER_PID 2>/dev/null || true
    fi
    wait $_MLIP_SERVER_PID 2>/dev/null || true
}

# 注册 EXIT trap 以确保清理
trap 'stop_mlip_server' EXIT

# ── 清理端口 ──
_cleanup_port || true

# ── 启动 MLIP server ──
echo "[RunG16GPU_orb] Starting MLIP server (METHOD=$METHOD, PORT=$MLIP_SERVER_PORT)..."
echo "[RunG16GPU_orb] Server log: $SERVER_LOG"

apptainer exec --fakeroot --nv \
    -B "$(pwd):$(pwd)" \
    -B "/share/home/CodeQ/Benchmark:/share/home/CodeQ/Benchmark" \
    --env "METHOD=$METHOD" \
    "$SIF_IMAGE" \
    bash -c "source /root/miniconda2/bin/activate orbmol2 && export CXX=g++ && python /share/home/CodeQ/Benchmark/bin/mlip_server.py" > "$SERVER_LOG" 2>&1 &

_MLIP_SERVER_PID=$!
echo "[RunG16GPU_orb] Server PID: $_MLIP_SERVER_PID"

# ── 等待 server 就绪 ──
echo "[RunG16GPU_orb] Waiting for MLIP server..."
waited=0
while [ $waited -lt 600 ]; do
    if ! kill -0 $_MLIP_SERVER_PID 2>/dev/null; then
        echo "[RunG16GPU_orb] ERROR: Server process died during startup."
        echo "Server log:"
        cat "$SERVER_LOG" 2>/dev/null
        exit 1
    fi
    if python -c "
import socket, re
host = '127.0.0.1'
try:
    with open('$SERVER_LOG', 'r') as f:
        for line in f:
            m = re.search(r'Listening on ([\d.]+):(\d+)', line)
            if m:
                host = m.group(1)
                break
except: pass
s = socket.socket(); s.settimeout(1); s.connect((host, $MLIP_SERVER_PORT)); s.close()
" 2>/dev/null; then
        break
    fi
    sleep 0.5
    waited=$((waited + 1))
done

if [ $waited -ge 600 ]; then
    echo "[RunG16GPU_orb] ERROR: Server startup timed out (300s)."
    echo "Server log:"
    cat "$SERVER_LOG" 2>/dev/null
    stop_mlip_server
    exit 1
fi
echo "[RunG16GPU_orb] Server ready."

# ── 依次运行所有 Gaussian 任务（串行：一个结束再跑下一个）──
source /share/apps/gaussian/g16-env.sh 2>/dev/null || true

failed=0
for GJF in "${GJF_LIST[@]}"; do
    dos2unix "$GJF" >/dev/null 2>&1 || true

    JobName=$(basename "$GJF" .gjf)
    SCRATCH="/tmp/g16_${JobName}_$$"
    mkdir -p "$SCRATCH"
    export GAUSS_SCRDIR="$SCRATCH"

    echo "========================================"
    echo "[RunG16GPU_orb] 开始任务: $GJF  $(date)"
    echo "GAUSS_SCRDIR=$GAUSS_SCRDIR" > "$JobName.log"

    # g16 同步执行：当前任务跑完才继续下一个
    if g16 < "$GJF" > "$JobName.log"; then
        rc=0
    else
        rc=$?
    fi
    date >> "$JobName.log"

    if [ "$rc" -eq 0 ]; then
        echo "[RunG16GPU_orb] 任务完成: $GJF -> $JobName.log  $(date)"
    else
        echo "[RunG16GPU_orb] 任务失败(rc=$rc): $GJF"
        failed=$((failed + 1))
    fi
    rm -rf "$SCRATCH"
done

echo "========================================"
if [ "$failed" -eq 0 ]; then
    echo "[RunG16GPU_orb] 全部 ${#GJF_LIST[@]} 个任务处理完成。"
    exit 0
else
    echo "[RunG16GPU_orb] 处理完成，其中 $failed/${#GJF_LIST[@]} 个任务失败。"
    exit 1
fi
