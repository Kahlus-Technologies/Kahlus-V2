#!/usr/bin/env bash
# One E1 run on one GPU: pretrain, embed the final checkpoint on the readout anchors, probe. Exits at once when done.json exists.
#   run_one.sh ARM SIZE CELL SEED LR MASK STEPS [NAME]      ARM = signal|latent, SIZE = 10m|50m|200m, CELL = r1|r4|r16|clean4|dirty4
set -euo pipefail
ARM=$1 SIZE=$2 CELL=$3 SEED=$4 LR=$5 MASK=$6 STEPS=$7 NAME=${8:-$ARM-$SIZE-$CELL-s$SEED}
REPO=${REPO:?} PY=${PY:?}
E1=${E1:-/shared/aayu/e1}
RUN=$E1/runs/$NAME
[[ -f $RUN/done.json ]] && exit 0
CKEVERY=1000; [[ $SIZE == 200m ]] && CKEVERY=300   # a checkpoint at least every 15 minutes
t0=$(date +%s)
"$PY" -m kahlus.train --shards "$E1/../fill-a/shards/shhs1-full" --out "$RUN" --size "$SIZE" --train-datasets shhs \
  --night-list "$E1/lists/$CELL.txt" --objective "$ARM" --head mlp --ecg-drop 0.25 --batch 512 --accum 1 --context 64 --horizon 4 \
  --schedule wsd --warmup 1000 --amp bf16 --lr "$LR" --mask-ratio "$MASK" --steps "$STEPS" --seed "$SEED" --ckpt-every "$CKEVERY" --workers 6
t1=$(date +%s)
CK=$(ls "$RUN"/ckpt-*.pt | sort | tail -n 1)
"$PY" "$REPO/scripts/e1_eval.py" embed --work "$E1" --ckpt "$CK" --out "$RUN/emb.npy" --workers 6
"$PY" "$REPO/scripts/e1_eval.py" probe --work "$E1" --emb "$RUN/emb.npy" --out "$RUN/probe.npz"
rm -f "$RUN/emb.npy"
t2=$(date +%s)
"$PY" - "$RUN" "$NAME" "$t1" "$t0" "$t2" "$STEPS" "$LR" "$MASK" <<'PYEOF'
import json, sys
run, name, t1, t0, t2, steps, lr, mask = sys.argv[1:]
last = [json.loads(x) for x in open(run + "/log.jsonl")][-1]
json.dump({"name": name, "steps": int(steps), "lr": float(lr), "mask_ratio": float(mask), "train_gpu_seconds": int(t1) - int(t0),
           "total_gpu_seconds": int(t2) - int(t0), "final_loss": last["loss"], "target_std": last.get("target_std")}, open(run + "/done.json", "w"))
PYEOF
