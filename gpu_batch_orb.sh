#!/bin/bash
# gpu_batch_orb.sh — orbmol_v2 版本的批量 XYZ 能量计算（通过 Apptainer + conda 环境）
#
# 用法:
#   ./gpu_batch_orb.sh [--method METHOD] [xyz_files_or_dirs...]
#
# 默认: 当前目录下所有 .xyz 文件，方法 orbmol_v2
#
# 示例:
#   ./gpu_batch_orb.sh *.xyz
#   ./gpu_batch_orb.sh --method orbmol residue.xyz ligand.xyz
#   ./gpu_batch_orb.sh /path/to/xyz_dir

METHOD="orbmol_v2"
CUSTOM_OUTPUT=""
ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --method|-m)
            METHOD="$2"
            shift 2
            ;;
        --output|-o)
            CUSTOM_OUTPUT="$2"
            shift 2
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

SIF_IMAGE="/share/home/CodeQ/docker/orbmol"
BATCH_SCRIPT="/share/home/CodeQ/Benchmark/bin/batch_xyz.py"
WORK_DIR="$PWD"

# 默认：当前目录所有 .xyz
if [[ ${#ARGS[@]} -eq 0 ]]; then
    ARGS=(*.xyz)
fi

echo "=== GPU Batch XYZ (orbmol) ==="
echo "方法:   $METHOD"
echo "文件数: ${#ARGS[@]}"
echo "镜像:   $SIF_IMAGE"
echo "====================="

OUTPUT_ARG=()
if [[ -n "$CUSTOM_OUTPUT" ]]; then
    OUTPUT_ARG=(--output "$CUSTOM_OUTPUT")
fi

apptainer exec --fakeroot --nv \
    -B "${WORK_DIR}:${WORK_DIR}" \
    -B "/share/home/CodeQ/.mlatom/models:/share/home/CodeQ/.mlatom/models" \
    -B "/share/home/CodeQ/Benchmark:/share/home/CodeQ/Benchmark" \
    "${SIF_IMAGE}" \
    bash -c "source /root/miniconda2/bin/activate orbmol2 && export CXX=g++ && python ${BATCH_SCRIPT} \
        --method ${METHOD} \
        ${OUTPUT_ARG[*]} \
        ${ARGS[*]}"
