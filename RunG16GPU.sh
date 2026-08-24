#!/bin/bash
# aRunG16.sh — 本地运行 Gaussian16 + MLIP GPU server (无需 PBS 调度器)
#
# 用法: ./aRunG16.sh input.gjf [METHOD]
#   METHOD 默认 mace_off24，可选 mace_omol, mace_polar
#
# 工作流:
#   1. 清理端口残留
#   2. 通过 Apptainer 启动 mlip_server.py（GPU server）
#   3. 等待 server 就绪
#   4. 运行 g16
#   5. 自动清理 server

set -e

if [[ $# -le 0 ]]; then
    echo -e "\033[32mUsage: $0 input.gjf [METHOD]\033[0m"
    echo -e "\033[33m  METHOD: mace_off24 (default), mace_omol, mace_polar\033[0m"
    exit 101
fi

dos2unix "$1" >/dev/null 2>&1 || true

# ── 参数 ──
GJF="$1"
export METHOD="${2:-mace_off24}"

Nprocs=$(head -10 "$GJF" | grep -i "nproc" | awk -F'=' '{print $2}')
Nmem=$(head -10 "$GJF" | grep -i "mem" | head -1 | awk -F'=' '{print $2}')
[[ -z "$Nprocs" ]] && Nprocs=1
[[ -z "$Nmem"   ]] && Nmem="${Nprocs}gb"

JobName=$(basename "$GJF" .gjf)
SCRATCH="/tmp/g16_${JobName}_$$"
mkdir -p "$SCRATCH"

# ── 引入 Apptainer 工具函数 ──
BIN_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$BIN_DIR/apptainer_utils.sh"

# 注册 EXIT trap 以确保清理
trap 'stop_mlip_server; rm -rf "$SCRATCH"' EXIT

# ── 启动 server ──
export MLIP_SERVER_PORT=15556
start_mlip_server "$BIN_DIR/mlip_server.py"

# ── 运行 Gaussian ──
source /share/apps/gaussian/g16-env.sh 2>/dev/null || true
export GAUSS_SCRDIR="$SCRATCH"
echo "GAUSS_SCRDIR=$GAUSS_SCRDIR" >> "$JobName.log"

echo "[aRunG16] Running Gaussian16..."
g16 < "$GJF" >> "$JobName.log"

date >> "$JobName.log"
echo "[aRunG16] Job finished. Output: $JobName.log"
