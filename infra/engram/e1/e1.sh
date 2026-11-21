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
  *) echo "unknown STAGE $STAGE" >&2; exit 2 ;;
esac
