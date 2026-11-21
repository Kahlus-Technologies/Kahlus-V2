#!/usr/bin/env bash
# Pilots A and B (docs/prereg/pilot_a.md, pilot_b.md) on the kept partial shards and the 10M checkpoint, at most 4 GPUs.
# Every stage is skipped when its output exists, so a rerun after a stop continues. Not a claim.
set -euo pipefail
unset CDPATH
REPO=$PWD
ROOT=${FILL_ROOT:-/shared/aayu}
W=$ROOT/pilot-ab
mkdir -p "$W"
PY=$(FILL_ROOT=$ROOT bash "$REPO/infra/engram/fill/setup_env.sh")
export PYTHONPATH=$REPO/src PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
P="$PY $REPO/scripts/pilot_ab.py"
# sklearn and joblib are needed by the CPU stages; setup_env.sh installs scikit-learn already.
NGPU=${NGPU:-4}
log() { printf '== %s %s\n' "$*" "$(date -u +%FT%TZ)"; }

# gpu_jobs <file of command lines>: one worker per GPU takes every NGPU-th line, so at most NGPU jobs run at once
gpu_jobs() {
  local f=$1 g p fail=0 pids=()
  for g in $(seq 0 $((NGPU - 1))); do
    ( awk -v g="$g" -v n="$NGPU" 'NR % n == g' "$f" | while IFS= read -r cmd; do CUDA_VISIBLE_DEVICES=$g bash -c "$cmd" || exit 1; done ) &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  return $fail
}

[[ -s $W/anchors.npz ]] || { log anchors; $P anchors --work "$W" --out "$W" --workers 60 | tee "$W/anchors.log"; }
cat "$W/anchors.log" 2>/dev/null | tail -12
# Gate G1: at least 100 held-out subjects with a 30 s label and 1,000 positive 30 s test labels, else stop before the GPU work
subjects=$(sed -n 's/^test subjects with a 30 s label: \([0-9]*\);.*/\1/p' "$W/anchors.log")
positives=$(sed -n 's/^  horizon  30s: positives fit [0-9]*, test \([0-9]*\)$/\1/p' "$W/anchors.log")
echo "G1: held-out subjects with a 30 s label $subjects, test positives at 30 s $positives"
(( subjects >= 100 && positives >= 1000 )) || { echo "PILOT_AB_G1_FAILED subjects=$subjects positives=$positives"; exit 3; }
[[ ${STAGE:-all} != anchors ]] || { echo PILOT_AB_ANCHORS_DONE; exit 0; }  # the CPU-only job stops here: no GPU is held for data prep

log embed
: > "$W/embed.jobs"
for set in a b; do for inp in asis perwindow; do
  [[ -s $W/emb-$set-pretrained-$inp-s0.npy ]] || echo "$P embed --work $W --set $set --init pretrained --input $inp --out $W/emb-$set-pretrained-$inp-s0.npy" >> "$W/embed.jobs"
  for s in 0 1 2; do
    [[ -s $W/emb-$set-random-$inp-s$s.npy ]] || echo "$P embed --work $W --set $set --init random --seed $s --input $inp --out $W/emb-$set-random-$inp-s$s.npy" >> "$W/embed.jobs"
  done
  [[ -s $W/bandpower-$set-$inp-s0.npy ]] || echo "$P embed --work $W --set $set --init bandpower --input $inp --out $W/bandpower-$set-$inp-s0.npy" >> "$W/embed.jobs"
done; done
[[ ! -s $W/embed.jobs ]] || gpu_jobs "$W/embed.jobs"

log probe  # the CPU probes run beside the fine-tunes, so the GPUs do not wait for them
{ [[ -s $W/probe.npz ]] || $P probe --work "$W" --jobs 16; } > "$W/probe.log" 2>&1 & probe_pid=$!
{ [[ -s $W/probe_b.npy ]] || $P probe-b --work "$W" --jobs 8; } > "$W/probe_b.log" 2>&1 & probe_b_pid=$!

log finetune
: > "$W/ft1.jobs"
for init in pretrained random; do for s in 0 1 2; do for lr in 1e-4 3e-4 1e-3; do
  echo "$P finetune --work $W --init $init --seed $s --lr $lr --out $W/ft-$init-s$s-lr$lr" >> "$W/ft1.jobs"
done; done; done
gpu_jobs "$W/ft1.jobs"
# the shift control runs at the learning rate that won for each pretrained seed (lowest tune log loss at 30 s)
: > "$W/ft2.jobs"
for s in 0 1 2; do
  lr=$($PY - "$W" "$s" <<'PYEOF'
import sys, numpy as np
from pathlib import Path
w, s = Path(sys.argv[1]), sys.argv[2]
print(min((float(np.load(d / "pred.npz")["tune_ll30"]), d.name.split("-lr")[1]) for d in w.glob(f"ft-pretrained-s{s}-lr*"))[1])
PYEOF
)
  echo "$P finetune --work $W --init pretrained --seed $s --lr $lr --shift 1 --out $W/ft-pretrained-s$s-shift" >> "$W/ft2.jobs"
done
gpu_jobs "$W/ft2.jobs"

wait "$probe_pid" || { cat "$W/probe.log"; exit 1; }
wait "$probe_b_pid" || { cat "$W/probe_b.log"; exit 1; }
tail -n 8 "$W/probe.log" "$W/probe_b.log"
log report
$P report --work "$W" | tee "$W/report.log"
for f in report.md per_subject.csv; do
  echo "PILOT_AB_FILE_BEGIN $f"; gzip -c "$W/$f" | base64 -w0; echo; echo "PILOT_AB_FILE_END $f"
done
echo PILOT_AB_DONE
