#!/usr/bin/env bash
# Launch single-process train.py and route every artifact (log, checkpoints,
# wandb files) under a fresh timestamped directory.
#
# Usage:
#   scripts/run_train.sh --config_path configs/b2d_overfit.yaml [other train.py args]
#
# A run directory is created at:
#   $B2D_TRAIN_ROOT/<YYYYmmdd_HHMMSS>_<config_basename>/
# where B2D_TRAIN_ROOT defaults to /home/sakoda/data/b2d_result/train.
#
# Env vars:
#   B2D_TRAIN_ROOT       (default: /home/sakoda/data/b2d_result/train)
#   CUDA_VISIBLE_DEVICES (default: 0)
#   MASTER_PORT          (default: 29501)

set -euo pipefail

cd "$(dirname "$0")/.."

# Pull --config_path out of the forwarded args to derive a run tag.
TAG=""
prev=""
for arg in "$@"; do
    if [[ "$prev" == "--config_path" ]]; then
        TAG="$(basename "$arg" .yaml)"
        break
    fi
    prev="$arg"
done
TAG="${TAG:-train}"

ROOT_DIR="${B2D_TRAIN_ROOT:-/home/sakoda/data/b2d_result/train}"
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
    "$@" \
    2>&1 | tee "${RUN_DIR}/train.log"
