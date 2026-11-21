#!/usr/bin/env bash
# Kahlus-V2 "Run 0" on one Engram node, as resumable stages: environment, the mirror gate, shards,
# then the pretraining sweep 10M -> 50M -> 200M on SHHS only, 8 GPUs with torchrun. Exploratory infrastructure,
# not a scientific claim. MESA is never downloaded and never trained on. Rerun after any stop: every stage whose
# outputs exist is skipped, and a training run resumes from its newest checkpoint.
#
# Runs inside the SkyPilot task fill_a.yaml. Everything it writes goes to $FILL on /shared. It prints the small
# summary files between FILL_SUMMARY_BEGIN/END markers, so the launching server can keep them from the log.
#
# Settings (environment): SIZES (default "10m": larger sizes wait for the captain's evidence gate), NGPU (default 4, the captain's budget; 8 only for the 200M burst),
# SHARDS_ONLY=1 (CPU-only job: download check and shards, no GPU held; fill_shards.yaml),
# REQUIRE_SHARDS=1 (fail instead of downloading when /shared holds no shards: /shared may be per node),
# STEPS_<SIZE> (optimizer steps for one size, for example STEPS_200M; default: 4 passes over the data),
# CKPT_EVERY_<SIZE> (steps between checkpoints, tuned so a checkpoint lands about every 15 minutes).
set -euo pipefail
unset CDPATH
REPO=$PWD
ROOT=${FILL_ROOT:-/shared/aayu}
FILL=$ROOT/fill-a
NSRR=$ROOT/nsrr            # SHHS-1 only: kahlus.nsrr rejects a subject that has two EDFs in one tree
SIZES=${SIZES:-"10m"}
NGPU=${NGPU:-4}
COMMIT=${FILL_COMMIT:-unknown}
mkdir -p "$FILL"/{logs,state,summaries,shards,runs} "$NSRR"
umask 077
chmod 700 "$ROOT" "$NSRR" 2>/dev/null || true
log() { printf '== %s %s\n' "$*" "$(date -u +%FT%TZ)"; }

if [[ ${SHARDS_ONLY:-0} != 1 ]]; then
  log "nvidia-smi"; nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
  [[ $(nvidia-smi --list-gpus | wc -l) -ge $NGPU ]] || { echo "fewer than $NGPU GPUs visible" >&2; exit 2; }
fi

# ---- environment (CUDA torch into /shared/aayu/venv, once) ----
PY=$(bash "$REPO/infra/engram/fill/setup_env.sh")
export PYTHONPATH=$REPO/src PYTHONUNBUFFERED=1
"$PY" -c 'import torch,kahlus; print("torch", torch.__version__, "gpus", torch.cuda.device_count(), kahlus.__file__)'

# ---- the mirror gate: no shards and no training unless /shared/aayu/nsrr holds every SHHS-1 file, MD5-verified ----
# nsrr_mirror.py (fill_nsrr.yaml) writes verified.txt and prints NSRR_MIRROR_DONE only after the full count.
BASE=$NSRR/shhs/polysomnography
if [[ ${REQUIRE_SHARDS:-0} != 1 ]]; then
  n_edf=$(ls "$BASE"/edfs/shhs1/*.edf 2>/dev/null | wc -l); n_xml=$(ls "$BASE"/annotations-events-nsrr/shhs1/*.xml 2>/dev/null | wc -l)
  n_part=$(find "$NSRR" -name '*.part' | wc -l)
  echo "mirror: edf=$n_edf xml=$n_xml part=$n_part"
  [[ $n_edf == 5793 && $n_xml == 5793 && $n_part == 0 ]] || { echo "the NSRR mirror is not complete: refusing to build shards" >&2; exit 4; }
fi
stage() {  # stage <name> <function>: skipped when state/<name>.ok exists
  local n=$1
  if [[ -f "$FILL/state/$n.ok" ]]; then log "skip $n"; return 0; fi
  log "start $n"; "$2"; echo "$COMMIT" > "$FILL/state/$n.ok"; log "done $n"
}
[[ ${REQUIRE_SHARDS:-0} != 1 || -s $FILL/shards/shhs1-full/index.json ]] || { echo "no shards on this node's /shared" >&2; exit 3; }

SHARDS=$FILL/shards/shhs1-full  # the first build (shards/shhs1) used 873 partial nights and is not used
build_shards() {
  rm -rf "$SHARDS.partial"
  NSRR_SHHS_ROOT=$NSRR/shhs/polysomnography "$PY" -m kahlus.shards nsrr --cohorts shhs --out "$SHARDS.partial" --jobs 24
  mv "$SHARDS.partial" "$SHARDS"
}
stage shards_full build_shards
"$PY" - "$SHARDS/index.json" <<'PYEOF'
import json, sys
i = json.load(open(sys.argv[1]))
print("shards:", len(i["nights"]), "nights,", sum(n["n"] for n in i["nights"]), "patches,", sorted({n["dataset"] for n in i["nights"]}))
PYEOF

[[ ${SHARDS_ONLY:-0} != 1 ]] || { echo FILL_SHARDS_DONE; exit 0; }  # the CPU-only data-prep job stops here

# ---- the sweep ----
declare -A BATCH=([10m]=256 [50m]=128 [200m]=64) LR=([10m]=6e-4 [50m]=4e-4 [200m]=2e-4)  # scripts/train_a100.sh values
declare -A CKPT=([10m]=${CKPT_EVERY_10M:-500} [50m]=${CKPT_EVERY_50M:-250} [200m]=${CKPT_EVERY_200M:-100})
export OMP_NUM_THREADS=4

sweep_point() {
  local size=$1 run=$FILL/runs/full/kahlus-$1 steps_var
  steps_var=STEPS_${size^^}
  "$PY" -m kahlus.run_env preflight --expect-gpus "$NGPU" > "$FILL/logs/preflight.json"
  "$PY" -m torch.distributed.run --standalone --nproc_per_node="$NGPU" -m kahlus.train \
    --shards "$SHARDS" --out "$run" --size "$size" --train-datasets shhs \
    ${!steps_var:+--steps "${!steps_var}"} --batch "${BATCH[$size]}" --lr "${LR[$size]}" --workers "${WORKERS:-6}" --amp bf16 --ckpt-every "${CKPT[$size]}"
  local ckpt; ckpt=$(find "$run" -name 'ckpt-*.pt' | sort | tail -n 1)
  "$PY" -m kahlus.forecast_bench --ckpt "$ckpt" --shards "$SHARDS" --device cuda --out "$run/benchmark.json"
  "$PY" -m kahlus.run_env summary "$run"
}

emit_summary() {  # small text records only: no checkpoints, no NSRR data
  local size=$1 run=$FILL/runs/full/kahlus-$1
  mkdir -p "$FILL/summaries/$size"
  cp "$run"/{summary.md,benchmark.json,environment.json} "$FILL/summaries/$size/"
  tail -n 40 "$run/log.jsonl" > "$FILL/summaries/$size/log-tail.jsonl"
  echo "FILL_SUMMARY_BEGIN $size"
  cat "$run/summary.md"; echo; cat "$run/benchmark.json"; echo; tail -n 3 "$run/log.jsonl"
  echo "FILL_SUMMARY_END $size"
}

for size in $SIZES; do
  t0=$(date +%s)
  if [[ -s $FILL/runs/full/kahlus-$size/summary.md && -s $FILL/runs/full/kahlus-$size/benchmark.json ]]; then log "skip sweep $size"; else
    log "start sweep $size"; sweep_point "$size"; log "done sweep $size"
    echo "$size wall_s=$(($(date +%s) - t0)) gpus=$NGPU" >> "$FILL/summaries/gpu-hours.txt"
  fi
  emit_summary "$size"
done

echo "FILL_A_DONE"
