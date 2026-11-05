#!/usr/bin/env bash
# Behavior check for refactors: rerun the cache-backed studies and diff every
# value against the committed artifacts. Usage: scripts/check_artifacts_unchanged.sh OUT CACHE_ROOT
set -u
out=${1:?out dir}
cache=${2:?dir holding bench/cache, beat/cache, clas/cache from repro_dev_slice.sh}
mesa=${MESA:-$HOME/data/nsrr/mesa}
shhs=${SHHS:-$HOME/data/nsrr/shhs}
mkdir -p "$out/beat" "$out/clas"
cp -R "$cache/beat/cache" "$out/beat/"
cp -R "$cache/clas/cache" "$out/clas/"
run() { name=$1; shift; if "$@" > "$out/$name.log" 2>&1; then echo "$name ok"; else echo "$name FAILED"; fi; }
run bench python scripts/run_heart_arousal_benchmark.py --mesa "$mesa" --shhs "$shhs" --out "$out/bench" --cache "$cache/bench/cache" --normalize causal
run conf  python scripts/run_conformal_alarm.py --cache "$cache/bench/cache" --out "$out/conf"
run beat  python scripts/run_beat_warning.py --mesa "$mesa" --shhs "$shhs" --out "$out/beat"
run clas  python scripts/run_clas_replay.py --mesa "$mesa" --shhs "$shhs" --out "$out/clas"
run mont  python scripts/run_montage_stress.py --mesa "$mesa" --shhs "$shhs" --out "$out/mont"
python3 - "$out" <<'E'
import json, math, sys
out = sys.argv[1]
def flat(d, p=""):
    o = {}
    if isinstance(d, dict):
        for k, v in d.items():
            o.update(flat(v, f"{p}/{k}"))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            o.update(flat(v, f"{p}[{i}]"))
    else:
        o[p] = d
    return o
def same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12) or (math.isnan(a) and math.isnan(b))
    return a == b
bad = 0
for n, a in [("bench", "heart_arousal_dev25_causal"), ("conf", "conformal_alarm_dev25"), ("beat", "beat_warning_dev25"),
             ("clas", "clas_replay_dev25"), ("mont", "montage_stress_dev25")]:
    x = flat(json.load(open(f"artifacts/{a}/report.json")))
    y = flat(json.load(open(f"{out}/{n}/report.json")))
    d = [k for k in set(x) | set(y) if not same(x.get(k), y.get(k))]
    bad += len(d)
    print(f"{n:6s} values {len(x):5d} differ {len(d)}", d[:3])
sys.exit(1 if bad else 0)
E
