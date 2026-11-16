#!/usr/bin/env bash
# One sweep point of issue #11 on the 8x A100 node: pretrain, then score the checkpoint.
#
#   scripts/train_a100.sh <10m|50m|200m|1b> <shards-dir> <run-root>
#   sbatch --gres=gpu:8 --cpus-per-task=64 scripts/train_a100.sh 10m /scratch/shards /scratch/runs
#
# Rerun the same command after a crash or a time limit: training resumes from the newest checkpoint.
# Every size sees the same data: the same shards, the same steps, and 256 windows per GPU per step.
# The batch, accumulation and learning-rate values below are starting points. Nobody has measured
# them on an A100 yet.
#
# Settings (environment):
#   NGPU            processes to start, one per GPU (default: every visible GPU)
#   TRAIN_DATASETS  default "mesa mros", the plan of issue #11
#   TEST_DATASET    default "shhs", never trained on
#   STEPS           optimizer steps (default 100000)
#   WORKERS         loader processes per GPU (default 6)
#   ALLOW_1B=1      needed for the 1b point: set it only after 10m to 200m show a slope
#   DRY_RUN=1       print the commands and run nothing
set -euo pipefail

if (($# != 3)); then
  echo "usage: scripts/train_a100.sh <10m|50m|200m|1b> <shards-dir> <run-root>" >&2
  exit 2
fi
SIZE=$1 SHARDS=$2 RUN_ROOT=$3
case "$SIZE" in
  10m) BATCH=256 ACCUM=1 LR=6e-4 ;;
  50m) BATCH=128 ACCUM=2 LR=4e-4 ;;
  200m) BATCH=64 ACCUM=4 LR=2e-4 ;;
  1b)
    BATCH=32 ACCUM=8 LR=1e-4
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
if [[ "$DRY_RUN" != 1 ]]; then
  [[ -f "$SHARDS/index.json" ]] || { echo "no shards at $SHARDS: build them with python -m kahlus.shards nsrr --out $SHARDS" >&2; exit 2; }
  NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}
  ((NGPU >= 1)) || { echo "no GPU visible" >&2; exit 2; }
fi
NGPU=${NGPU:-8}
RUN="$RUN_ROOT/kahlus-$SIZE"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}  # NGPU ranks and their loaders share the CPUs

run() {
  printf '+'; printf ' %q' "$@"; printf '\n'
  [[ "$DRY_RUN" == 1 ]] || "$@"
}

# shellcheck disable=SC2086  # TRAIN_DATASETS is a space-separated list on purpose
run torchrun --standalone --nproc-per-node="$NGPU" -m kahlus.train \
  --shards "$SHARDS" --out "$RUN" --size "$SIZE" --train-datasets ${TRAIN_DATASETS:-mesa mros} \
  --steps "${STEPS:-100000}" --batch "$BATCH" --accum "$ACCUM" --lr "$LR" --workers "${WORKERS:-6}" --amp bf16

CKPT=$(find "$RUN" -name 'ckpt-*.pt' 2>/dev/null | sort | tail -n 1 || true)
run python -m kahlus.forecast_bench --ckpt "${CKPT:-$RUN/ckpt-<last>.pt}" --shards "$SHARDS" \
  --test-dataset "${TEST_DATASET:-shhs}" --device cuda --out "$RUN/benchmark.json"
