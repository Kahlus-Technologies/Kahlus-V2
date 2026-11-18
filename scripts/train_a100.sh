#!/usr/bin/env bash
# One sweep point of issue #11 on the 8x A100 node: preflight, pretrain, score the checkpoint, summarize.
#
#   scripts/train_a100.sh <10m|50m|200m|1b> <shards-dir> <run-root>
#   DOCKER=1 scripts/train_a100.sh 10m /raid/scratch/$USER/shards /raid/scratch/$USER/runs
#
# The script needs no conda and no sbatch. It uses the host Python when that can import torch and
# kahlus. With DOCKER=1 it runs itself inside a PyTorch container instead (needs the NVIDIA container
# toolkit, and network access to pull the image and to pip install this repo).
#
# Rerun the same command after a crash or a time limit: training resumes from the newest checkpoint.
# Every size sees the same data: the same shards, the same steps, and an effective batch that the
# trainer derives from the data size. BATCH below is only the per-GPU micro-batch that fits in memory.
# The micro-batch and learning-rate values are starting points. Nobody has measured them on an A100 yet.
#
# Settings (environment):
#   NGPU            processes to start, one per GPU (default 8). The preflight fails unless exactly
#                   this many GPUs are visible. Use CUDA_VISIBLE_DEVICES to hide GPUs, never an index here.
#   TRAIN_DATASETS  default "shhs" (MrOS cannot be loaded yet; MESA stays sealed)
#   TEST_DATASET    optional, a dataset never trained on; default none, so only subject-held-out is scored
#   STEPS           optimizer steps (default: 4 passes over the training data, worked out by the trainer)
#   WORKERS         loader processes per GPU (default 6)
#   ALLOW_1B=1      needed for the 1b point: set it only after 10m to 200m show a slope
#   DOCKER=1        run inside DOCKER_IMAGE (default pytorch/pytorch:2.9.1-cuda12.6-cudnn9-devel)
#   DRY_RUN=1       print the commands and run nothing
set -euo pipefail

if (($# != 3)); then
  echo "usage: scripts/train_a100.sh <10m|50m|200m|1b> <shards-dir> <run-root>" >&2
  exit 2
fi
SIZE=$1 SHARDS=$2 RUN_ROOT=$3
case "$SIZE" in
  10m) BATCH=256 LR=6e-4 ;;
  50m) BATCH=128 LR=4e-4 ;;
  200m) BATCH=64 LR=2e-4 ;;
  1b)
    BATCH=32 LR=1e-4
    if [[ "${ALLOW_1B:-0}" != 1 ]]; then
      echo "1b is gated: issue #11 allows it only after 10m to 200m show a slope. Set ALLOW_1B=1 then." >&2
      exit 2
    fi
    ;;
  *) echo "unknown size: $SIZE" >&2; exit 2 ;;
esac
if [[ "$RUN_ROOT" != /* || "$SHARDS" != /* ]]; then
  echo "shards-dir and run-root must be absolute paths on persistent storage" >&2
  exit 2
fi

DRY_RUN=${DRY_RUN:-0}
NGPU=${NGPU:-8}
RUN="$RUN_ROOT/kahlus-$SIZE"

run() {
  printf '+'; printf ' %q' "$@"; printf '\n'
  [[ "$DRY_RUN" == 1 ]] || "$@"
}

if [[ "${DOCKER:-0}" == 1 ]]; then
  REPO=$(cd "$(dirname "$0")/.." && pwd)
  IMAGE=${DOCKER_IMAGE:-pytorch/pytorch:2.9.1-cuda12.6-cudnn9-devel}
  [[ "$DRY_RUN" == 1 ]] || mkdir -p "$RUN_ROOT"
  # --ipc=host gives the loader workers the host's shared memory, so no --shm-size is needed.
  # The container sees every GPU; CUDA_VISIBLE_DEVICES and NGPU select among them, as on the host.
  run docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$REPO":/workspace/repo -v "$SHARDS":"$SHARDS" -v "$RUN_ROOT":"$RUN_ROOT" -w /workspace/repo \
    -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0=/workspace/repo \
    -e DOCKER=0 -e KAHLUS_DOCKER_IMAGE="$IMAGE" -e NGPU="$NGPU" -e STEPS -e WORKERS -e TRAIN_DATASETS -e TEST_DATASET \
    -e ALLOW_1B -e CUDA_VISIBLE_DEVICES -e KAHLUS_MESA \
    "$IMAGE" bash -c 'pip install -q -e . && exec scripts/train_a100.sh "$@"' train_a100 "$SIZE" "$SHARDS" "$RUN_ROOT"
  exit 0
fi

if [[ "$DRY_RUN" != 1 ]]; then
  [[ -f "$SHARDS/index.json" ]] || { echo "no shards at $SHARDS: build them with python -m kahlus.shards nsrr --out $SHARDS" >&2; exit 2; }
  python -c 'import torch, kahlus' 2>/dev/null || { echo "this Python cannot import torch and kahlus: pip install -e '.[train]', or use DOCKER=1" >&2; exit 2; }
fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}  # NGPU ranks and their loaders share the CPUs

# Stops here unless exactly NGPU GPUs are visible. It prints their names and the software versions.
run python -m kahlus.run_env preflight --expect-gpus "$NGPU"

# Each process takes the GPU that torchrun names in LOCAL_RANK. No GPU index is written anywhere.
# torchrun runs as `python -m torch.distributed.run`, so its workers use this python even when torch
# comes from the system site-packages of a venv (the torchrun script there names the system python).
# shellcheck disable=SC2086  # TRAIN_DATASETS is a space-separated list on purpose
run python -m torch.distributed.run --standalone --nproc_per_node="$NGPU" -m kahlus.train \
  --shards "$SHARDS" --out "$RUN" --size "$SIZE" --train-datasets ${TRAIN_DATASETS:-shhs} \
  ${STEPS:+--steps "$STEPS"} --batch "$BATCH" --lr "$LR" --workers "${WORKERS:-6}" --amp bf16

CKPT=$(find "$RUN" -name 'ckpt-*.pt' 2>/dev/null | sort | tail -n 1 || true)
run python -m kahlus.forecast_bench --ckpt "${CKPT:-$RUN/ckpt-<last>.pt}" --shards "$SHARDS" \
  ${TEST_DATASET:+--test-dataset "$TEST_DATASET"} --device cuda --out "$RUN/benchmark.json"

# summary.md states that this run is infrastructure validation, not a scientific claim.
run python -m kahlus.run_env summary "$RUN"
