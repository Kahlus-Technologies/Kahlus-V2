#!/usr/bin/env bash
# Reproduce the dev-slice studies into a new run directory. Any failed step stops the
# run and the script exits nonzero.
#
# Usage: scripts/repro_dev_slice.sh [RUN_ID]
#
# The run writes OUT_BASE/RUN_ID (default artifacts/runs/<utc time>-<commit>) and refuses
# a directory that already exists, so no step ever sees another run's cache. Inside it:
#   run.json         commit, data roots, settings, start time
#   discovery/       per-cohort discovery report and, with SHHS_LIMIT, the subset ID list
#   logs/<step>.log  one log per step
#   steps.tsv        exit code, wall, user and system seconds and peak RSS per step
#   run_end.json     ok or failed, the failed step, end time
#
# Settings (environment):
#   NSRR_ROOT        parent of mesa/ and shhs/ (default ~/data/nsrr)
#   NSRR_MESA_ROOT, NSRR_SHHS_ROOT
#                    per-cohort roots; split (edfs/ + annotations/) or co-located layout
#   BOOTSTRAP        claim (default, 2000 resamples) or smoke (300)
#   SHHS_LIMIT       use N SHHS nights picked by scripts/nsrr_discover.py (default: all)
#   SUBSET_SEED      seed for that pick (default 0)
#   OUT_BASE         parent of run directories (default artifacts/runs)
#   STEPS            steps to run (default: census bench beat conf conf_prior clas mont elhr).
#                    bench_raw and bench_night (--normalize none / night diagnostics) are opt-in.
#   JOBS             steps run at once (default 1; more needs bash 4.3+)
#   THREADS          OpenMP/BLAS threads per step (default: library default)
#   PYTHON           interpreter (default python3)
# KAHLUS_MESA must be unset or dev: this script never makes the final MESA look.
set -euo pipefail

die() { echo "repro_dev_slice: $*" >&2; exit 2; }

repo=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo"

case ${KAHLUS_MESA:-dev} in
  dev) ;;
  *) die "KAHLUS_MESA=${KAHLUS_MESA} refused; the dev-slice run is dev mode only" ;;
esac
PY=${PYTHON:-python3}
BOOTSTRAP=${BOOTSTRAP:-claim}
case $BOOTSTRAP in claim|smoke) ;; *) die "BOOTSTRAP must be claim or smoke, got $BOOTSTRAP" ;; esac
JOBS=${JOBS:-1}
[[ $JOBS =~ ^[1-9][0-9]*$ ]] || die "JOBS must be a positive integer"
if (( JOBS > 1 )) && (( BASH_VERSINFO[0] * 100 + BASH_VERSINFO[1] < 403 )); then
  die "JOBS>1 needs bash 4.3 or newer (wait -n); this is $BASH_VERSION"
fi
SUBSET_SEED=${SUBSET_SEED:-0}
[[ -z ${SHHS_LIMIT:-} || $SHHS_LIMIT =~ ^[1-9][0-9]*$ ]] || die "SHHS_LIMIT must be a positive integer"
read -r -a steps <<< "${STEPS-census bench beat conf conf_prior clas mont elhr}"
(( ${#steps[@]} )) || die "STEPS is empty"
phase1=() phase2=()
for s in "${steps[@]}"; do
  case $s in
    census|bench|beat|clas|mont|elhr) phase1+=("$s") ;;
    conf|conf_prior|bench_raw|bench_night) phase2+=("$s") ;;  # read this run's bench cache
    *) die "unknown step '$s'" ;;
  esac
done
if (( ${#phase2[@]} )) && [[ " ${phase1[*]-} " != *" bench "* ]]; then
  die "${phase2[*]} read the bench cache from this run, so STEPS must include bench"
fi

nsrr_root=${NSRR_ROOT:-$HOME/data/nsrr}
mesa=${NSRR_MESA_ROOT:-$nsrr_root/mesa}
shhs=${NSRR_SHHS_ROOT:-$nsrr_root/shhs}
[[ -d $mesa ]] || die "MESA root $mesa is not a directory (set NSRR_MESA_ROOT or NSRR_ROOT)"
[[ -d $shhs ]] || die "SHHS root $shhs is not a directory (set NSRR_SHHS_ROOT or NSRR_ROOT)"
mesa=$(cd "$mesa" && pwd -P)
shhs=$(cd "$shhs" && pwd -P)

commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)
dirty=false
[[ -n $(git status --porcelain --untracked-files=no 2>/dev/null) ]] && dirty=true
run_id=${1:-$(date -u +%Y%m%dT%H%M%SZ)-${commit:0:9}}
[[ $run_id =~ ^[A-Za-z0-9._-]+$ ]] || die "RUN_ID may use letters, digits, dot, dash and underscore only"
out_base=${OUT_BASE:-artifacts/runs}
mkdir -p "$out_base"
run_dir=$(cd "$out_base" && pwd -P)/$run_id
mkdir "$run_dir" 2>/dev/null || die "$run_dir already exists; every run needs a new directory"
mkdir "$run_dir/logs"

export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n ${THREADS:-} ]]; then
  export OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS
fi

utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
json_str() { local s=${1//\\/\\\\}; s=${s//\"/\\\"}; printf '"%s"' "$s"; }
{
  printf '{\n'
  printf '  "run_id": %s,\n' "$(json_str "$run_id")"
  printf '  "commit": %s,\n  "dirty": %s,\n' "$(json_str "$commit")" "$dirty"
  printf '  "start_utc": %s,\n' "$(json_str "$(utc)")"
  printf '  "host": %s,\n' "$(json_str "$(hostname)")"
  printf '  "mesa_root": %s,\n  "shhs_root": %s,\n' "$(json_str "$mesa")" "$(json_str "$shhs")"
  printf '  "bootstrap": %s,\n  "normalize": "causal",\n' "$(json_str "$BOOTSTRAP")"
  printf '  "shhs_limit": %s,\n  "subset_seed": %s,\n' "$(json_str "${SHHS_LIMIT:-all}")" "$(json_str "$SUBSET_SEED")"
  printf '  "steps": %s,\n  "jobs": %s,\n' "$(json_str "${steps[*]}")" "$JOBS"
  printf '  "threads": %s,\n  "python": %s,\n' "$(json_str "${THREADS:-default}")" "$(json_str "$PY")"
  printf '  "kahlus_mesa": "dev"\n}\n'
} > "$run_dir/run.json"

failed=""
finish() {
  local rc=$?
  local status=ok
  (( rc == 0 )) || status=failed
  printf '{"status": "%s", "exit": %d, "failed_step": %s, "end_utc": "%s"}\n' \
    "$status" "$rc" "$(json_str "$failed")" "$(utc)" > "$run_dir/run_end.json"
  (( rc == 0 )) && echo "done: $run_dir" || echo "FAILED (${failed:-setup}): see $run_dir/logs" >&2
}
trap finish EXIT

# GNU time gives peak RSS; elsewhere only wall time is recorded.
if /usr/bin/time -f '%e' -o /dev/null true 2>/dev/null; then gnu_time=1; else gnu_time=0; fi
printf 'step\texit\twall_s\tuser_s\tsys_s\tmax_rss_kb\tstart_utc\tend_utc\n' > "$run_dir/steps.tsv"

step_cmd() {
  case $1 in
    census)      cmd=("$PY" scripts/nsrr_arousal_census.py "$mesa" "$shhs" --out "$run_dir/census") ;;
    bench)       cmd=("$PY" scripts/run_heart_arousal_benchmark.py --mesa "$mesa" --shhs "$shhs" --out "$run_dir/bench"
                      --cache "$run_dir/bench/cache" --bootstrap "$BOOTSTRAP" --normalize causal) ;;
    bench_raw)   cmd=("$PY" scripts/run_heart_arousal_benchmark.py --mesa "$mesa" --shhs "$shhs" --out "$run_dir/bench_raw"
                      --cache "$run_dir/bench/cache" --bootstrap "$BOOTSTRAP" --normalize none) ;;
    bench_night) cmd=("$PY" scripts/run_heart_arousal_benchmark.py --mesa "$mesa" --shhs "$shhs" --out "$run_dir/bench_night"
                      --cache "$run_dir/bench/cache" --bootstrap "$BOOTSTRAP" --normalize night) ;;
    beat)        cmd=("$PY" scripts/run_beat_warning.py --mesa "$mesa" --shhs "$shhs" --out "$run_dir/beat" --bootstrap "$BOOTSTRAP") ;;
    conf)        cmd=("$PY" scripts/run_conformal_alarm.py --cache "$run_dir/bench/cache" --out "$run_dir/conf") ;;
    conf_prior)  cmd=("$PY" scripts/run_conformal_alarm.py --cache "$run_dir/bench/cache" --out "$run_dir/conf_prior" --model prior) ;;
    clas)        cmd=("$PY" scripts/run_clas_replay.py --mesa "$mesa" --shhs "$shhs" --out "$run_dir/clas") ;;
    mont)        cmd=("$PY" scripts/run_montage_stress.py --mesa "$mesa" --shhs "$shhs" --out "$run_dir/mont") ;;
    elhr)        cmd=("$PY" scripts/event_locked_hr.py "$mesa:EKG" "$shhs:ECG" --out "$run_dir/elhr") ;;
    *) return 1 ;;
  esac
}

run_step() {
  local name=$1 log="$run_dir/logs/$1.log" tfile="$run_dir/logs/.$1.time" start rc=0 t0 wall usr sys rss
  local -a cmd
  step_cmd "$name"
  start=$(utc)
  t0=$(date +%s)
  echo "start $name"
  if (( gnu_time )); then
    /usr/bin/time -f $'%e\t%U\t%S\t%M' -o "$tfile" "${cmd[@]}" > "$log" 2>&1 || rc=$?
    IFS=$'\t' read -r wall usr sys rss < <(tail -n 1 "$tfile") || true
    rm -f "$tfile"
  else
    "${cmd[@]}" > "$log" 2>&1 || rc=$?
    wall=$(( $(date +%s) - t0 )) usr=NA sys=NA rss=NA
  fi
  printf '%s\t%d\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$rc" "$wall" "$usr" "$sys" "$rss" "$start" "$(utc)" >> "$run_dir/steps.tsv"
  if (( rc == 0 )); then echo "ok    $name (${wall}s)"; else echo "FAIL  $name exit $rc (see $log)" >&2; fi
  return "$rc"
}

# With JOBS>1 each step runs in its own process group, so stopping the run (a failed
# step, timeout or Ctrl-C) stops every running step and its children.
kill_steps() {
  local p
  for p in $(jobs -p); do kill -TERM -- "-$p" 2>/dev/null || kill -TERM "$p" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap '[[ -n $failed ]] || failed=signal; kill_steps; exit 1' TERM INT
stop_on_failure() {
  local bad
  bad=$(awk -F'\t' 'NR > 1 && $2 != 0 { print $1; exit }' "$run_dir/steps.tsv")
  [[ -z $bad ]] && return 0
  failed=$bad
  kill_steps
  exit 1
}
running() { jobs -rp | wc -l; }

# Steps in a phase are independent. conf, conf_prior and the bench diagnostics read
# the bench cache, so they wait for phase 1.
run_phase() {
  local s
  if (( JOBS == 1 )); then
    for s in "$@"; do run_step "$s" || { failed=$s; exit 1; }; done
    return 0
  fi
  set -m
  for s in "$@"; do
    while (( $(running) >= JOBS )); do wait -n || true; stop_on_failure; done
    run_step "$s" &
  done
  while (( $(running) > 0 )); do wait -n || true; stop_on_failure; done
  wait || true
  stop_on_failure
  set +m
}

echo "run $run_id: mesa=$mesa shhs=$shhs bootstrap=$BOOTSTRAP jobs=$JOBS threads=${THREADS:-default}"
subset=()
[[ -n ${SHHS_LIMIT:-} ]] && subset=(--limit "$SHHS_LIMIT" --seed "$SUBSET_SEED")
"$PY" scripts/nsrr_discover.py --cohort mesa --root "$mesa" --out "$run_dir/discovery" \
  > "$run_dir/logs/discover_mesa.log" 2>&1 || { failed=discover_mesa; exit 1; }
"$PY" scripts/nsrr_discover.py --cohort shhs --root "$shhs" --out "$run_dir/discovery" ${subset[@]+"${subset[@]}"} \
  > "$run_dir/logs/discover_shhs.log" 2>&1 || { failed=discover_shhs; exit 1; }
if [[ -n ${SHHS_LIMIT:-} ]]; then
  export NSRR_SHHS_IDS="$run_dir/discovery/ids_shhs.txt"
  [[ -s $NSRR_SHHS_IDS ]] || { failed=discover_shhs; exit 1; }
fi

run_phase ${phase1[@]+"${phase1[@]}"}
run_phase ${phase2[@]+"${phase2[@]}"}
