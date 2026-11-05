#!/usr/bin/env bash
# Reproduce every dev-slice study from a fresh clone into a new output dir.
# Usage: scripts/repro_dev_slice.sh OUT_DIR [MESA_DIR] [SHHS_DIR]
set -u
out=${1:?output dir}
mesa=${2:-$HOME/data/nsrr/mesa}
shhs=${3:-$HOME/data/nsrr/shhs}
mkdir -p "$out"
run() {
  name=$1; shift
  if "$@" > "$out/$name.log" 2>&1; then echo "$name ok"; else echo "$name FAILED (see $out/$name.log)"; fi
}
run census python scripts/nsrr_arousal_census.py "$mesa" "$shhs" --out "$out/census"
run bench  python scripts/run_heart_arousal_benchmark.py --mesa "$mesa" --shhs "$shhs" --out "$out/bench" --normalize causal
run bench_raw   python scripts/run_heart_arousal_benchmark.py --mesa "$mesa" --shhs "$shhs" --out "$out/bench_raw" --cache "$out/bench/cache" --normalize none
run bench_night python scripts/run_heart_arousal_benchmark.py --mesa "$mesa" --shhs "$shhs" --out "$out/bench_night" --cache "$out/bench/cache" --normalize night
run beat   python scripts/run_beat_warning.py --mesa "$mesa" --shhs "$shhs" --out "$out/beat"
run conf   python scripts/run_conformal_alarm.py --cache "$out/bench/cache" --out "$out/conf"
run conf_prior python scripts/run_conformal_alarm.py --cache "$out/bench/cache" --out "$out/conf_prior" --model prior
run clas   python scripts/run_clas_replay.py --mesa "$mesa" --shhs "$shhs" --out "$out/clas"
run mont   python scripts/run_montage_stress.py --mesa "$mesa" --shhs "$shhs" --out "$out/mont"
run elhr   python scripts/event_locked_hr.py "$mesa:EKG" "$shhs:ECG" --out "$out/elhr"
