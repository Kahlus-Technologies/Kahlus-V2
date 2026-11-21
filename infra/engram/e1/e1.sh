#!/usr/bin/env bash
# E1 stages (docs/prereg/e1.md). STAGE=cpu: night lists and quality gate GQ, then the readout anchors. STAGE=g0: gate G0 on the newest
# full-data 10M checkpoint (1 GPU). Outputs on /shared/aayu/e1. Every stage is skipped when its output exists.
set -euo pipefail
unset CDPATH
REPO=$PWD
ROOT=${FILL_ROOT:-/shared/aayu}
E1=$ROOT/e1
mkdir -p "$E1"
PY=$(FILL_ROOT=$ROOT bash "$REPO/infra/engram/fill/setup_env.sh")
export PYTHONPATH=$REPO/src PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
log() { printf '== %s %s\n' "$*" "$(date -u +%FT%TZ)"; }
case ${STAGE:?} in
  cpu)
    [[ -s $E1/gq.json ]] || { log data; $PY "$REPO/scripts/e1_data.py" --out "$E1" --workers 60 | tee "$E1/data.log"; }
    cat "$E1/data.log" 2>/dev/null
    [[ -s $E1/anchors.npz ]] || { log anchors; $PY "$REPO/scripts/e1_eval.py" anchors --work "$E1" --workers 60 | tee "$E1/anchors.log"; }
    cat "$E1/anchors.log" 2>/dev/null
    echo E1_CPU_DONE ;;
  g0)
    CK=$(ls "$ROOT"/fill-a/runs/full/kahlus-10m/ckpt-*.pt | sort | tail -n 1)
    log "g0 on $CK"
    [[ -s $E1/g0.json ]] || $PY "$REPO/scripts/e1_eval.py" g0 --work "$E1" --ckpt "$CK" --workers 16 | tee "$E1/g0.log"
    cat "$E1/g0.json"
    echo E1_G0_DONE ;;
  pilots)
    # Step 8: pilots at r1, seed 0, on 4 GPUs (the captain's cap). Phase 1: 10M at full length, mask x lr x arm. Phase 2: 50M and 200M at 25% of the
    # steps, lr x arm, with the mask chosen at 10M for that arm. Runs skip themselves when done.json exists, so a relaunch continues.
    RUN1="env REPO=$REPO PY=$PY E1=$E1 bash $REPO/infra/engram/e1/run_one.sh"
    J1=$E1/pilots1.jobs; : > "$J1"
    for arm in signal latent; do
      steps=19200; [[ $arm == latent ]] && steps=14400
      for mask in 0.3 0.6; do for lr in 1e-4 3e-4 1e-3; do
        echo "$RUN1 $arm 10m r1 0 $lr $mask $steps $arm-10m-r1-s0-m$mask-lr$lr" >> "$J1"
      done; done
    done
    log "pilots phase 1"; $PY "$REPO/scripts/e1_queue.py" "$J1" --gpus "${NGPU:-4}"
    $PY "$REPO/scripts/e1_eval.py" g0tune --work "$E1"
    $PY "$REPO/scripts/e1_freeze.py" 10m --work "$E1"
    J2=$E1/pilots2.jobs; : > "$J2"
    for size in 200m 50m; do for arm in signal latent; do
      mask=$($PY -c "import json;print(json.load(open('$E1/frozen.json'))['$arm']['mask'])")
      steps=4800; [[ $arm == latent ]] && steps=3600
      for lr in 1e-4 3e-4 1e-3; do echo "$RUN1 $arm $size r1 0 $lr $mask $steps $arm-$size-r1-s0-pilot-lr$lr" >> "$J2"; done
    done; done
    log "pilots phase 2"; $PY "$REPO/scripts/e1_queue.py" "$J2" --gpus "${NGPU:-4}"
    $PY "$REPO/scripts/e1_freeze.py" big --work "$E1" | tee "$E1/frozen.log"
    echo E1_PILOTS_DONE ;;
  rung1|rung2)
    RUN1="env REPO=$REPO PY=$PY E1=$E1 bash $REPO/infra/engram/e1/run_one.sh"
    J=$E1/$STAGE.jobs
    $PY "$REPO/scripts/e1_jobs.py" "$STAGE" --work "$E1" --run1 "$RUN1" > "$J"
    log "$STAGE: $(wc -l < "$J") runs to do"
    $PY "$REPO/scripts/e1_queue.py" "$J" --gpus "${NGPU:-4}"
    [[ $STAGE == rung1 ]] && $PY "$REPO/scripts/e1_eval.py" k2 --work "$E1"
    [[ $STAGE == rung2 ]] && { $PY "$REPO/scripts/e1_eval.py" report --work "$E1"; cat "$E1/report.md"; }
    echo "E1_${STAGE^^}_DONE" ;;
  *) echo "unknown STAGE $STAGE" >&2; exit 2 ;;
esac
