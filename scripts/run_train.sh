#!/usr/bin/env bash
# Launch single-process train.py and route every artifact (log, checkpoints,
# wandb files) under a fresh timestamped directory.
#
# Usage:
#   scripts/run_train.sh --root-dir /path/to/results \
#                        --config_path configs/b2d_finetune.yaml \
#                        [other train.py args]
#
# A run directory is created at:
#   <root-dir>/<YYYYmmdd_HHMMSS>_<config_basename>/
#
# Env vars:
#   CUDA_VISIBLE_DEVICES (default: 0)
#   MASTER_PORT          (default: 29501)

set -euo pipefail

cd "$(dirname "$0")/.."

# Parse out --root-dir (consumed by this script) and --config_path (used to
# derive a run tag); forward everything else to train.py.
ROOT_DIR=""
TAG=""
FORWARD_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --root-dir)
            ROOT_DIR="$2"
            shift 2
            ;;
        --config_path)
            TAG="$(basename "$2" .yaml)"
            FORWARD_ARGS+=("$1" "$2")
            shift 2
            ;;
        *)
            FORWARD_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "${ROOT_DIR}" ]]; then
    echo "error: --root-dir is required" >&2
    exit 2
fi
TAG="${TAG:-train}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT_DIR}/${STAMP}_${TAG}"
mkdir -p "${RUN_DIR}"

echo "run dir: ${RUN_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
RANK=0 \
LOCAL_RANK=0 \
WORLD_SIZE=1 \
MASTER_ADDR=localhost \
MASTER_PORT="${MASTER_PORT:-29501}" \
uv run python train.py \
    --logdir "${RUN_DIR}" \
    --wandb-save-dir "${RUN_DIR}" \
    "${FORWARD_ARGS[@]}" \
    2>&1 | tee "${RUN_DIR}/train.log"
