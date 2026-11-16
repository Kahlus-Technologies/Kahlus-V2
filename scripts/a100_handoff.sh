#!/usr/bin/env bash
# The A100 handoff run, in stages. After a failure, run the same command again: every stage whose
# outputs already verify for this commit is skipped. A failure packs the evidence zip before it exits.
#
#   scripts/a100_handoff.sh                       # every stage, in order
#   scripts/a100_handoff.sh verify smoke          # only the named stages, still in the order below
#   DOCKER=1 scripts/a100_handoff.sh              # Python stages in a PyTorch container
#
# Stages:
#   verify     package and repo checksums, and the commit (always runs)
#   preflight  stops unless exactly NGPU GPUs are visible; writes the environment record (always runs)
#   smoke      synthetic data: a tiny model on 1 GPU (CPU when there is none), then 20 steps on NGPU GPUs
#   fetch      NSRR dev slice with your own ~/.nsrr_token: MESA + SHHS-1 annotations, 25 EDFs each (~7 GB)
#   devslice   scripts/repro_dev_slice.sh on that slice
#   train10m   SHHS patch shards, then scripts/train_a100.sh 10m (4 passes over the training data)
#   evidence   the zip to send back, from scripts/package_evidence.sh (always runs)
#
# Settings (environment):
#   WORK       outputs, Python environment and data. Default /raid/scratch/$USER/kahlus-a100 when that
#              can be made, else ~/kahlus-a100
#   NSRR_ROOT  NSRR data (default $WORK/nsrr)
#   NGPU       GPUs that the preflight expects and that training uses (default 8)
#   DOCKER=1   run the Python stages in DOCKER_IMAGE (default pytorch/pytorch:2.9.1-cuda12.6-cudnn9-devel)
#   The called scripts read their own settings: BOOTSTRAP, JOBS, THREADS (devslice), STEPS, WORKERS (train10m).
set -euo pipefail

REPO=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)
PKG=$(dirname "$REPO")
STAGES=(verify preflight smoke fetch devslice train10m evidence)
die() { echo "a100_handoff: $*" >&2; exit 1; }

want=("$@")
((${#want[@]})) || want=("${STAGES[@]}")
for s in "${want[@]}"; do
  [[ " ${STAGES[*]} " == *" $s "* ]] || die "unknown stage '$s' (stages: ${STAGES[*]})"
done

if [[ -z ${WORK:-} ]]; then
  WORK=$HOME/kahlus-a100
  scratch=/raid/scratch/${USER:-$(id -un)}/kahlus-a100
  [[ -d /raid/scratch ]] && mkdir -p "$scratch" 2>/dev/null && WORK=$scratch
fi
mkdir -p "$WORK/logs" "$WORK/state" "$WORK/home"
WORK=$(cd "$WORK" && pwd -P)
export NSRR_ROOT=${NSRR_ROOT:-$WORK/nsrr}
export NGPU=${NGPU:-8}
DOCKER=${DOCKER:-0}
IMAGE=${DOCKER_IMAGE:-pytorch/pytorch:2.9.1-cuda12.6-cudnn9-devel}
if [[ $DOCKER == 1 ]]; then VENV=$WORK/venv-docker; else VENV=$WORK/venv-host; fi
COMMIT=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || cat "$PKG/COMMIT_HASH.txt" 2>/dev/null || echo unknown)
SHORT=${COMMIT:0:12}
DEVSLICE=$WORK/devslice/handoff-$SHORT
RUN10M=$WORK/runs/kahlus-10m
EVIDENCE=$WORK/kahlus-evidence-$SHORT.zip

# Runs a command with the stage Python first on PATH, on this host or in the container.
in_env() {
  if [[ $DOCKER == 1 ]]; then
    local mounts=(-v "$REPO:$REPO" -v "$WORK:$WORK")
    [[ $NSRR_ROOT == "$WORK"/* ]] || mounts+=(-v "$NSRR_ROOT:$NSRR_ROOT")
    docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
      --user "$(id -u):$(id -g)" "${mounts[@]}" -w "$REPO" -e HOME="$WORK/home" -e PYTHONUNBUFFERED=1 \
      -e KAHLUS_DOCKER_IMAGE="$IMAGE" -e NSRR_ROOT -e NGPU -e CUDA_VISIBLE_DEVICES -e KAHLUS_MESA \
      -e BOOTSTRAP -e JOBS -e THREADS -e STEPS -e WORKERS -e OMP_NUM_THREADS -e PIP_INDEX_URL -e PIP_EXTRA_INDEX_URL \
      "$IMAGE" bash -c 'export PATH="$0/bin:$PATH"; exec "$@"' "$VENV" "$@"
  else
    (export PATH="$VENV/bin:$PATH"; "$@")
  fi
}

# A Python that has torch, the study libraries, and kahlus from this repo. Built once in $VENV.
ensure_env() {
  local check='import sys, torch, mne, sklearn, kahlus; sys.exit(not kahlus.__file__.startswith(sys.argv[1]))'
  in_env python -c "$check" "$REPO/src/" 2>/dev/null && return 0
  echo "building the Python environment in $VENV (pip needs network)"
  in_env python3 -m venv --system-site-packages "$VENV" || die "python3 -m venv failed: install python3-venv, or use DOCKER=1"
  in_env "$VENV/bin/python" -m pip install -q -e "$REPO[train]"
  in_env python -c "$check" "$REPO/src/" || die "the environment in $VENV cannot import torch, mne, sklearn and kahlus"
}

stage_verify() {
  if [[ -f $PKG/SHA256SUMS && -f $PKG/COMMIT_HASH.txt ]]; then
    (cd "$PKG" && sha256sum --check --strict --quiet SHA256SUMS) || die "package files do not match $PKG/SHA256SUMS"
    echo "ok: package files match $PKG/SHA256SUMS"
  fi
  if [[ -f $REPO/HANDOFF_SHA256SUMS ]]; then
    (cd "$REPO" && sha256sum --check --strict --quiet HANDOFF_SHA256SUMS) || die "repo files do not match HANDOFF_SHA256SUMS"
    echo "ok: repo files match $REPO/HANDOFF_SHA256SUMS"
  else
    echo "no HANDOFF_SHA256SUMS: this is a git checkout, not the handoff package"
  fi
  if git -C "$REPO" rev-parse HEAD >/dev/null 2>&1; then
    [[ -z $(git -C "$REPO" status --porcelain --untracked-files=no) ]] || die "tracked files changed: see git -C $REPO status"
    if [[ -f $PKG/COMMIT_HASH.txt && $(<"$PKG/COMMIT_HASH.txt") != "$COMMIT" ]]; then
      die "checkout is at $COMMIT, but COMMIT_HASH.txt says $(<"$PKG/COMMIT_HASH.txt")"
    fi
    echo "ok: clean checkout"
  fi
  printf 'commit %s\nverified %s\n' "$COMMIT" "$(date -u +%FT%TZ)" | tee "$WORK/verify.txt"
}

stage_preflight() {
  mkdir -p "$WORK/preflight"
  command -v nvidia-smi >/dev/null || die "nvidia-smi not found: this stage needs the GPU node"
  nvidia-smi | tee "$WORK/preflight/nvidia-smi.txt"
  [[ $DOCKER != 1 ]] || docker version > "$WORK/preflight/docker-version.txt"
  ensure_env
  in_env python -m kahlus.run_env preflight --expect-gpus "$NGPU" > "$WORK/preflight/environment.json"
  cat "$WORK/preflight/environment.json"
}

done_smoke() { [[ -s $WORK/smoke/run/summary.md && -s $WORK/smoke/run/benchmark.json ]]; }
stage_smoke() {
  local d=$WORK/smoke dev gpus
  rm -rf "$d"
  ensure_env
  in_env python -m kahlus.shards synthetic --out "$d/shards"
  dev=$(in_env python -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')
  in_env python -m kahlus.train --shards "$d/shards" --out "$d/run" --size tiny --train-datasets synth_a \
    --steps 200 --warmup 10 --context 8 --horizon 2 --workers 0
  in_env python -m kahlus.forecast_bench --ckpt "$d/run/ckpt-00000200.pt" --shards "$d/shards" \
    --test-dataset synth_b --device "$dev" --out "$d/run/benchmark.json"
  in_env python -m kahlus.run_env summary "$d/run"
  gpus=$(in_env python -c 'import torch; print(torch.cuda.device_count())')
  if ((NGPU > 1 && gpus >= NGPU)); then  # NCCL across all GPUs, before hours of fetch and training
    in_env python -m torch.distributed.run --standalone --nproc_per_node="$NGPU" -m kahlus.train --shards "$d/shards" --out "$d/ddp" \
      --size tiny --train-datasets synth_a --steps 20 --warmup 2 --context 8 --horizon 2 --workers 0 --amp bf16
    in_env python -m kahlus.run_env summary "$d/ddp"
  else
    echo "skipped the $NGPU-GPU part: $gpus GPU visible"
  fi
}

# Prints each NSRR file on the fetch lists that is missing or empty.
fetch_missing() {
  local ds list f
  for ds in mesa shhs; do
    for list in "$NSRR_ROOT/$ds/xml_list.txt" "$NSRR_ROOT/$ds/edf_list.txt"; do
      [[ -s $list ]] || { echo "$list"; continue; }
      while IFS= read -r f; do [[ -s $NSRR_ROOT/$ds/${f##*/} ]] || echo "$NSRR_ROOT/$ds/${f##*/}"; done < "$list"
    done
  done
}
done_fetch() { [[ -z $(fetch_missing) ]]; }
stage_fetch() {
  [[ -s $HOME/.nsrr_token ]] || die "no ~/.nsrr_token: put your sleepdata.org token there. It stays on this machine"
  mkdir -p "$NSRR_ROOT"
  "$REPO/scripts/nsrr_fetch_slice.sh"
  local missing
  missing=$(fetch_missing | wc -l)
  ((missing == 0)) || die "$missing NSRR files are missing after the fetch: run again to retry them"
}

done_devslice() { grep -q '"status": "ok"' "$DEVSLICE/run_end.json" 2>/dev/null; }
stage_devslice() {
  ensure_env
  [[ ! -e $DEVSLICE ]] || mv "$DEVSLICE" "$DEVSLICE.failed-$(date +%s)"  # repro_dev_slice.sh needs a new directory
  in_env env OUT_BASE="$WORK/devslice" PYTHON=python "$REPO/scripts/repro_dev_slice.sh" "${DEVSLICE##*/}"
}

done_train10m() { [[ -s $RUN10M/summary.md && -s $RUN10M/benchmark.json ]]; }
stage_train10m() {
  ensure_env
  if [[ ! -s $WORK/shards/index.json ]]; then
    rm -rf "$WORK/shards.partial"
    in_env python -m kahlus.shards nsrr --cohorts shhs --out "$WORK/shards.partial" --jobs 8
    mv "$WORK/shards.partial" "$WORK/shards"
  fi
  in_env env DOCKER=0 "$REPO/scripts/train_a100.sh" 10m "$WORK/shards" "$WORK/runs"
}

stage_evidence() { "$REPO/scripts/package_evidence.sh" "$WORK" "$EVIDENCE"; }

current=""
on_exit() {
  local rc=$?
  ((rc != 0)) && [[ -n $current ]] || return 0
  echo "FAILED at stage $current (exit $rc). Log: $WORK/logs/$current.log" >&2
  [[ $current == evidence ]] || stage_evidence >&2 || true
  echo "Send $EVIDENCE back. After a fix, run the same command again: finished stages are skipped." >&2
}
trap on_exit EXIT

echo "commit $COMMIT | WORK=$WORK | NSRR_ROOT=$NSRR_ROOT | NGPU=$NGPU | DOCKER=$DOCKER | stages: ${want[*]}"
for s in "${STAGES[@]}"; do
  [[ " ${want[*]} " == *" $s "* ]] || continue
  current=$s
  if declare -F "done_$s" >/dev/null && [[ $(cat "$WORK/state/$s.ok" 2>/dev/null) == "$COMMIT" ]] && "done_$s"; then
    echo "skip $s: its outputs verify"
    continue
  fi
  rm -f "$WORK/state/$s.ok"
  echo "== $s $(date -u +%FT%TZ)" | tee -a "$WORK/logs/$s.log"
  "stage_$s" 2>&1 | tee -a "$WORK/logs/$s.log"
  echo "$COMMIT" > "$WORK/state/$s.ok"
done
current=""
[[ " ${want[*]} " != *" evidence "* ]] || echo "Send this file back: $EVIDENCE"
