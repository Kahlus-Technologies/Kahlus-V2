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
#   fetch      NSRR data with your own ~/.nsrr_token: all MESA + SHHS-1 annotations, the 25 MESA EDFs of the
#              dev slice and every SHHS-1 EDF. With SEIZEIT2=1 also the public SeizeIT2 seizure dataset
#              (OpenNeuro ds005873, about 126 GB) into $WORK/seizeit2. Stops first when the disk is too small
#   devslice   scripts/repro_dev_slice.sh on the dev slice: the 25 MESA EDFs and the first 25 SHHS-1 EDFs
#   train10m   patch shards of every SHHS-1 night, then scripts/train_a100.sh 10m (4 passes over the training data)
#   evidence   the zip to send back, from scripts/package_evidence.sh (always runs)
#
# Settings (environment):
#   WORK       outputs, Python environment and data. Default /raid/scratch/$USER/kahlus-a100 when that
#              can be made, else ~/kahlus-a100
#   NSRR_ROOT  NSRR data (default $WORK/nsrr)
#   N_SHHS_EDF SHHS-1 EDFs to fetch (default 0: all of them; 25 gives only the dev slice)
#   FETCH_NEED_GB  free disk that the NSRR fetch needs, less what NSRR_ROOT holds already (default 1000;
#              MESA + SHHS are about 741 GB, and the shards need room too)
#   SEIZEIT2=1 fetch SeizeIT2 too (no account needed; needs 130 GB free at WORK)
#   (SeizeIT2 is fetched only: no stage reads it yet)
#   NGPU       GPUs that the preflight expects and that training uses (default 8)
#   DOCKER=1   run the Python stages in DOCKER_IMAGE (default pytorch/pytorch:2.9.1-cuda12.6-cudnn9-devel)
#   The called scripts read their own settings: BOOTSTRAP, JOBS, THREADS (devslice), STEPS, WORKERS (train10m).
set -euo pipefail
unset CDPATH

REPO=$(cd -- "$(dirname -- "$0")/.." && pwd -P)
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
WORK=$(realpath -m "$WORK")
[[ $WORK != / && "$(realpath "$HOME")/" != "$WORK"/* ]] || die "WORK=$WORK holds your home folder: use a folder of its own"
mkdir -p "$WORK/logs" "$WORK/state" "$WORK/home"
mkdir -p "${NSRR_ROOT:=$WORK/nsrr}"
NSRR_ROOT=$(cd "$NSRR_ROOT" && pwd -P)  # repro_dev_slice.sh and docker -v need an absolute path
export NSRR_ROOT
export NGPU=${NGPU:-8}
export N_SHHS_EDF=${N_SHHS_EDF:-0} N_EDF=${N_EDF:-25}
DOCKER=${DOCKER:-0}
IMAGE=${DOCKER_IMAGE:-pytorch/pytorch:2.9.1-cuda12.6-cudnn9-devel}
if [[ $DOCKER == 1 ]]; then VENV=$WORK/venv-docker; else VENV=$WORK/venv-host; fi
COMMIT=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || cat "$PKG/COMMIT_HASH.txt" 2>/dev/null || echo unknown)
SHORT=${COMMIT:0:12}
DEVSLICE=$WORK/devslice/handoff-$SHORT
TRAIN_KEY="$COMMIT n_shhs_edf=$N_SHHS_EDF ngpu=$NGPU docker=$DOCKER steps=${STEPS:-auto} workers=${WORKERS:-6} train_datasets=${TRAIN_DATASETS:-shhs}"
SETTINGS="$TRAIN_KEY n_edf=$N_EDF bootstrap=${BOOTSTRAP:-claim} seizeit2=${SEIZEIT2:-0}"
SHARDS=$WORK/shards/$SHORT-n$N_SHHS_EDF  # per commit and SHHS-1 night count: other settings never reuse these shards
RUN10M=$WORK/runs/$SHORT-$(printf %s "$TRAIN_KEY" | sha256sum | cut -c1-8)/kahlus-10m  # per setting: other settings never resume or replace this run

# The settings that change a stage's outputs: a finished stage is skipped only when these are unchanged.
stage_key() {
  case $1 in
    smoke) echo "$COMMIT ngpu=$NGPU docker=$DOCKER" ;;
    fetch) echo "$COMMIT n_edf=$N_EDF n_shhs_edf=$N_SHHS_EDF seizeit2=${SEIZEIT2:-0}" ;;
    devslice) echo "$COMMIT n_edf=$N_EDF bootstrap=${BOOTSTRAP:-claim} docker=$DOCKER" ;;
    train10m) echo "$TRAIN_KEY" ;;
    *) echo "$COMMIT" ;;
  esac
}
EVIDENCE=$WORK/kahlus-evidence-$SHORT.zip
SEIZEIT2_DIR=$WORK/seizeit2

# Runs a command with the stage Python first on PATH, on this host or in the container.
in_env() {
  if [[ $DOCKER == 1 ]]; then
    local mounts=(-v "$REPO:$REPO" -v "$WORK:$WORK")
    [[ $NSRR_ROOT == "$WORK"/* ]] || mounts+=(-v "$NSRR_ROOT:$NSRR_ROOT")
    docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
      --user "$(id -u):$(id -g)" "${mounts[@]}" -w "$REPO" -e HOME="$WORK/home" -e PYTHONUNBUFFERED=1 \
      -e KAHLUS_DOCKER_IMAGE="$IMAGE" -e NSRR_ROOT -e NGPU -e CUDA_VISIBLE_DEVICES -e KAHLUS_MESA \
      -e BOOTSTRAP -e JOBS -e THREADS -e STEPS -e WORKERS -e TRAIN_DATASETS -e OMP_NUM_THREADS -e PIP_INDEX_URL -e PIP_EXTRA_INDEX_URL \
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
  local changed
  if [[ -f $PKG/SHA256SUMS && -f $PKG/COMMIT_HASH.txt ]]; then
    # --ignore-missing: deleting the tarball after unpacking is fine. A changed file still fails.
    (cd "$PKG" && sha256sum --check --strict --quiet --ignore-missing SHA256SUMS) || die "package files do not match $PKG/SHA256SUMS"
    echo "ok: package files match $PKG/SHA256SUMS"
  elif [[ -f $REPO/HANDOFF_SHA256SUMS ]]; then
    die "keep kahlus-v2 in the unpacked package folder, beside SHA256SUMS and COMMIT_HASH.txt"
  fi
  if [[ -f $REPO/HANDOFF_SHA256SUMS ]]; then
    (cd "$REPO" && sha256sum --check --strict --quiet HANDOFF_SHA256SUMS) || die "repo files do not match HANDOFF_SHA256SUMS"
    echo "ok: repo files match $REPO/HANDOFF_SHA256SUMS"
  else
    echo "no HANDOFF_SHA256SUMS: this is a git checkout, not the handoff package"
  fi
  if git -C "$REPO" rev-parse HEAD >/dev/null 2>&1; then
    # Untracked files count too: a stray src/sklearn.py would change what runs.
    changed=$(git -C "$REPO" status --porcelain | grep -vx '?? HANDOFF_SHA256SUMS' || true)
    [[ -z $changed ]] || die "files changed or added since checkout:"$'\n'"$changed"
    if [[ -f $PKG/COMMIT_HASH.txt && $(<"$PKG/COMMIT_HASH.txt") != "$COMMIT" ]]; then
      die "checkout is at $COMMIT, but COMMIT_HASH.txt says $(<"$PKG/COMMIT_HASH.txt")"
    fi
    echo "ok: clean checkout"
  fi
  [[ $COMMIT != unknown ]] || die "cannot tell the commit: no git checkout and no COMMIT_HASH.txt"
  printf 'commit %s\nverified %s\n' "$COMMIT" "$(date -u +%FT%TZ)" | tee "$WORK/verify.txt"
}

stage_preflight() {
  mkdir -p "$WORK/preflight" "$NSRR_ROOT"
  df -hP "$WORK" "$NSRR_ROOT" | tee "$WORK/preflight/disk.txt"  # the fetch needs FETCH_NEED_GB (default 1000) of it
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
    --test-dataset synth_b --gru-steps 300 --device "$dev" --out "$d/run/benchmark.json"  # a short GRU: this checks plumbing
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

# Prints each NSRR file on the fetch lists that is missing, empty or not what its name says (for
# example an HTML error page saved as an EDF). An EDF starts with "0" and 7 spaces.
fetch_missing() {
  local ds list f path
  for ds in mesa shhs; do
    for list in "$NSRR_ROOT/$ds/xml_list.txt" "$NSRR_ROOT/$ds/edf_list.txt"; do
      [[ -s $list ]] || { echo "$list"; continue; }
      while IFS= read -r f; do
        path=$NSRR_ROOT/$ds/${f##*/}
        case $path in
          *.edf) [[ -s $path && $(head -c 8 "$path") == "0       " ]] || echo "$path" ;;
          *) [[ -s $path && $(head -c 512 "$path") == *PSGAnnotation* ]] || echo "$path" ;;
        esac
      done < "$list"
    done
  done
}
done_fetch() { [[ -z $(fetch_missing) ]] && [[ ${SEIZEIT2:-0} != 1 || -f $SEIZEIT2_DIR/.fetch-complete ]]; }

# Stops unless the disk under a folder has room for a download, counting what the folder holds already.
need_space() {
  local dir=$1 gb=$2 what=$3 have_kb free_kb
  mkdir -p "$dir"
  have_kb=$(du -sk "$dir" | cut -f1)
  free_kb=$(df -Pk "$dir" | awk 'NR == 2 { print $4 }')
  ((free_kb + have_kb >= gb * 1048576)) || die "$what needs about $gb GB at $dir. It holds $((have_kb / 1048576)) GB" \
    "and has $((free_kb / 1048576)) GB free. Set WORK or NSRR_ROOT to a larger disk"
}

stage_fetch() {
  [[ -s $HOME/.nsrr_token ]] || die "no ~/.nsrr_token: put your sleepdata.org token there. It stays on this machine"
  need_space "$NSRR_ROOT" "${FETCH_NEED_GB:-1000}" "The NSRR fetch (MESA + SHHS are about 741 GB, and the shards need room too; FETCH_NEED_GB sets the figure)"
  [[ ${SEIZEIT2:-0} != 1 ]] || need_space "$SEIZEIT2_DIR" 130 "SeizeIT2 (OpenNeuro ds005873, about 126 GB)"
  fetch_missing | while IFS= read -r f; do [[ $f == *_list.txt ]] || rm -f "$f"; done  # bad files download again
  N_SHHS_EDF=${N_SHHS_EDF:-0} "$REPO/scripts/nsrr_fetch_slice.sh"
  local missing
  missing=$(fetch_missing | wc -l)
  ((missing == 0)) || die "$missing NSRR files are missing or not valid after the fetch: run again to retry them"
  if [[ ${SEIZEIT2:-0} == 1 ]]; then
    ensure_env
    in_env aws --version >/dev/null 2>&1 || in_env python -m pip install -q awscli
    # A public bucket: no account and no token. sync resumes and skips files that are already complete.
    in_env aws s3 sync --no-sign-request --only-show-errors s3://openneuro.org/ds005873 "$SEIZEIT2_DIR"
    touch "$SEIZEIT2_DIR/.fetch-complete"
  fi
}

# SHHS-1 root of links: every annotation and the first N EDFs on the fetch list, even when NSRR_ROOT holds
# the whole cohort.
shhs_view() {
  local view=$1 n=$2 list=$NSRR_ROOT/shhs/edf_list.txt
  rm -rf "$view"
  mkdir -p "$view"
  find "$NSRR_ROOT/shhs" -maxdepth 1 -name '*.xml' -exec ln -s -t "$view" {} +
  if [[ -s $list ]]; then sed 's#.*/##' "$list"; else find "$NSRR_ROOT/shhs" -maxdepth 1 -name '*.edf' -printf '%f\n' | sort; fi |
    sed -n "1,${n}p" | while IFS= read -r f; do ln -s "$NSRR_ROOT/shhs/$f" "$view/$f"; done
}

done_devslice() { grep -q '"status": "ok"' "$DEVSLICE/run_end.json" 2>/dev/null; }
stage_devslice() {
  local view=$WORK/devslice-shhs
  ensure_env
  shhs_view "$view" "${N_EDF:-25}"
  if [[ -e $DEVSLICE ]]; then  # repro_dev_slice.sh needs a new directory
    if done_devslice; then mv "$DEVSLICE" "$DEVSLICE.superseded-$(date +%s)"; else mv "$DEVSLICE" "$DEVSLICE.failed-$(date +%s)"; fi
  fi
  in_env env -u STEPS OUT_BASE="$WORK/devslice" NSRR_SHHS_ROOT="$view" PYTHON=python "$REPO/scripts/repro_dev_slice.sh" "${DEVSLICE##*/}"
}

done_train10m() { [[ -s $RUN10M/summary.md && -s $RUN10M/benchmark.json ]]; }
stage_train10m() {
  ensure_env
  if [[ ! -s $SHARDS/index.json ]]; then
    local root=()
    if ((N_SHHS_EDF > 0)); then  # nsrr shards reads every EDF on disk: give it only the first N
      shhs_view "$WORK/shards-shhs-view" "$N_SHHS_EDF"
      root=(NSRR_SHHS_ROOT="$WORK/shards-shhs-view")
    fi
    rm -rf "$SHARDS.partial"
    in_env env "${root[@]}" python -m kahlus.shards nsrr --cohorts shhs --out "$SHARDS.partial" --jobs 32
    mv "$SHARDS.partial" "$SHARDS"
  fi
  in_env env DOCKER=0 "$REPO/scripts/train_a100.sh" 10m "$SHARDS" "${RUN10M%/*}"
}

stage_evidence() {
  rm -f "$EVIDENCE"
  printf 'settings: %s\ncurrent 10M run: %s\n' "$SETTINGS" "${RUN10M#"$WORK"/}" > "$WORK/logs/run-settings.txt"
  "$REPO/scripts/package_evidence.sh" "$WORK" "$EVIDENCE"
}

current=""
on_exit() {
  local rc=$?
  set +e  # after a dropped SSH session, writing to the terminal fails; the zip must still be packed
  ((rc != 0)) && [[ -n $current ]] || return 0
  echo "FAILED at stage $current (exit $rc). Log: $WORK/logs/$current.log" >&2
  [[ $current == evidence ]] || stage_evidence >&2 || true
  if [[ -s $EVIDENCE ]]; then
    echo "Send $EVIDENCE back. After a fix, run the same command again: finished stages are skipped." >&2
  else
    echo "The evidence zip failed too. Send the folder $WORK/logs back." >&2
  fi
}
trap on_exit EXIT
# Ctrl-C, kill or a dropped SSH session still packs the evidence zip. Run inside tmux or nohup to keep going.
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

echo "commit $COMMIT | WORK=$WORK | NSRR_ROOT=$NSRR_ROOT | NGPU=$NGPU | DOCKER=$DOCKER | stages: ${want[*]}"
for s in "${STAGES[@]}"; do
  [[ " ${want[*]} " == *" $s "* ]] || continue
  current=$s
  if declare -F "done_$s" >/dev/null && [[ $(cat "$WORK/state/$s.ok" 2>/dev/null) == "$(stage_key "$s")" ]] && "done_$s"; then
    echo "skip $s: its outputs verify"
    continue
  fi
  rm -f "$WORK/state/$s.ok"
  echo "== $s $(date -u +%FT%TZ)" | tee -a "$WORK/logs/$s.log"
  "stage_$s" 2>&1 | tee -a "$WORK/logs/$s.log"
  stage_key "$s" > "$WORK/state/$s.ok"
done
current=""
[[ " ${want[*]} " != *" evidence "* ]] || echo "Send this file back: $EVIDENCE"
