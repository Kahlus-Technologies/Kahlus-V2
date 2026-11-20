#!/usr/bin/env bash
# Kahlus-V2 "Run 0" on one Engram node, as resumable stages: environment, SHHS-1 and SHHS-2 download, shards,
# then the pretraining sweep 10M -> 50M -> 200M on SHHS only, 8 GPUs with torchrun. Exploratory infrastructure,
# not a scientific claim. MESA is never downloaded and never trained on. Rerun after any stop: every stage whose
# outputs exist is skipped, and a training run resumes from its newest checkpoint.
#
# Runs inside the SkyPilot task fill_a.yaml. Everything it writes goes to $FILL on /shared. It prints the small
# summary files between FILL_SUMMARY_BEGIN/END markers, so the launching server can keep them from the log.
#
# Settings (environment): SIZES (default "10m 50m 200m"), NGPU (default 4, the captain's budget; 8 only for the 200M burst),
# REQUIRE_SHARDS=1 (fail instead of downloading when /shared holds no shards: /shared may be per node), SHHS2 (1: also download SHHS-2 EDFs),
# STEPS_<SIZE> (optimizer steps for one size, for example STEPS_200M; default: 4 passes over the data),
# CKPT_EVERY_<SIZE> (steps between checkpoints, tuned so a checkpoint lands about every 15 minutes).
set -euo pipefail
unset CDPATH
REPO=$PWD
ROOT=${FILL_ROOT:-/shared/aayu}
FILL=$ROOT/fill-a
NSRR=$ROOT/nsrr            # SHHS-1 only: kahlus.nsrr rejects a subject that has two EDFs in one tree
NSRR2=$ROOT/nsrr-shhs2     # SHHS-2 EDFs, a separate tree on purpose; no repo code reads it yet
SIZES=${SIZES:-"10m 50m 200m"}
NGPU=${NGPU:-4}
COMMIT=${FILL_COMMIT:-unknown}
mkdir -p "$FILL"/{logs,state,summaries,shards,runs} "$NSRR" "$NSRR2"
umask 077
chmod 700 "$ROOT" "$NSRR" "$NSRR2" 2>/dev/null || true
log() { printf '== %s %s\n' "$*" "$(date -u +%FT%TZ)"; }

log "nvidia-smi"; nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
[[ $(nvidia-smi --list-gpus | wc -l) -ge $NGPU ]] || { echo "fewer than $NGPU GPUs visible" >&2; exit 2; }

# ---- environment (CUDA torch into /shared/aayu/venv, once) ----
PY=$(bash "$REPO/infra/engram/fill/setup_env.sh")
export PYTHONPATH=$REPO/src PYTHONUNBUFFERED=1
"$PY" -c 'import torch,kahlus; print("torch", torch.__version__, "gpus", torch.cuda.device_count(), kahlus.__file__)'

# ---- NSRR download. The token is a SkyPilot secret in $NSRR_TOKEN; never printed, never on a command line ----
export SSL_CERT_FILE=$HOME/.nsrr-ca.pem  # system CAs plus the Sectigo DV R36 intermediate (sleepdata.org sends an incomplete chain)
RUBY=$(bash "$REPO/infra/engram/fill/setup_ruby.sh")

fetch_cohort() {  # fetch_cohort <tree> <shhs1|shhs2> : annotations, then ten EDF streams split by the last digit of the ID
  local tree=$1 wave=$2 d p pids=()
  ( cd "$tree" && "$RUBY" "$REPO/infra/engram/fill/nsrr_fetch.rb" "shhs/polysomnography/annotations-events-nsrr/$wave" )
  for d in 0 1 2 3 4 5 6 7 8 9; do
    ( cd "$tree" && "$RUBY" "$REPO/infra/engram/fill/nsrr_fetch.rb" "shhs/polysomnography/edfs/$wave" "^$wave-[0-9]{5}${d}\.edf\$" ) \
      > "$FILL/logs/fetch-$wave-$d.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || { echo "fetch stream failed for $wave: see $FILL/logs/fetch-$wave-*.log" >&2; return 1; }; done
}

stage() {  # stage <name> <function>: skipped when state/<name>.ok exists for this commit
  local n=$1
  if [[ $(cat "$FILL/state/$n.ok" 2>/dev/null) == "$COMMIT" ]]; then log "skip $n"; return 0; fi
  log "start $n"; "$2"; echo "$COMMIT" > "$FILL/state/$n.ok"; log "done $n"
}

fetch_shhs1() { fetch_cohort "$NSRR" shhs1; du -sh "$NSRR/shhs"; }
fetch_shhs2() { fetch_cohort "$NSRR2" shhs2; du -sh "$NSRR2/shhs"; }
[[ ${REQUIRE_SHARDS:-0} != 1 || -s $FILL/shards/shhs1/index.json ]] || { echo "no shards on this node's /shared" >&2; exit 3; }
if [[ ${REQUIRE_SHARDS:-0} != 1 ]]; then stage fetch_shhs1 fetch_shhs1; fi

# SHHS-2 downloads in the background while the shards build: it needs network, the shards need CPUs.
if [[ ${SHHS2:-1} == 1 && ${REQUIRE_SHARDS:-0} != 1 ]]; then (stage fetch_shhs2 fetch_shhs2) > "$FILL/logs/fetch_shhs2.log" 2>&1 & shhs2_pid=$!; fi

SHARDS=$FILL/shards/shhs1
build_shards() {
  rm -rf "$SHARDS.partial"
  NSRR_SHHS_ROOT=$NSRR/shhs/polysomnography "$PY" -m kahlus.shards nsrr --cohorts shhs --out "$SHARDS.partial" --jobs 24
  mv "$SHARDS.partial" "$SHARDS"
}
stage shards build_shards
"$PY" - "$SHARDS/index.json" <<'PYEOF'
import json, sys
i = json.load(open(sys.argv[1]))
print("shards:", len(i["nights"]), "nights,", sum(n["n"] for n in i["nights"]), "patches,", sorted({n["dataset"] for n in i["nights"]}))
PYEOF

# ---- the sweep ----
declare -A BATCH=([10m]=256 [50m]=128 [200m]=64) LR=([10m]=6e-4 [50m]=4e-4 [200m]=2e-4)  # scripts/train_a100.sh values
declare -A CKPT=([10m]=${CKPT_EVERY_10M:-500} [50m]=${CKPT_EVERY_50M:-250} [200m]=${CKPT_EVERY_200M:-100})
export OMP_NUM_THREADS=4

sweep_point() {
  local size=$1 run=$FILL/runs/kahlus-$1 steps_var
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
  local size=$1 run=$FILL/runs/kahlus-$1
  mkdir -p "$FILL/summaries/$size"
  cp "$run"/{summary.md,benchmark.json,environment.json} "$FILL/summaries/$size/"
  tail -n 40 "$run/log.jsonl" > "$FILL/summaries/$size/log-tail.jsonl"
  echo "FILL_SUMMARY_BEGIN $size"
  cat "$run/summary.md"; echo; cat "$run/benchmark.json"; echo; tail -n 3 "$run/log.jsonl"
  echo "FILL_SUMMARY_END $size"
}

for size in $SIZES; do
  t0=$(date +%s)
  if [[ -s $FILL/runs/kahlus-$size/summary.md && -s $FILL/runs/kahlus-$size/benchmark.json ]]; then log "skip sweep $size"; else
    log "start sweep $size"; sweep_point "$size"; log "done sweep $size"
    echo "$size wall_s=$(($(date +%s) - t0)) gpus=$NGPU" >> "$FILL/summaries/gpu-hours.txt"
  fi
  emit_summary "$size"
done

[[ -z ${shhs2_pid:-} ]] || wait "$shhs2_pid"
echo "FILL_A_DONE"
