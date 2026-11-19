#!/usr/bin/env bash
# The seizure-alarm CPU benchmark at full scale on one Engram node, no GPU: the repo's own baseline, run with
# scripts/benchmark.py unchanged (every fold and seed) on CHB-MIT, Siena and SeizeIT2 within-dataset, and cross-site
# CHB-MIT -> Siena and CHB-MIT -> SeizeIT2. Open data only (PhysioNet and OpenNeuro, no account), downloaded here
# with the repo's own fetch scripts. Runs from a snapshot of the seizure-alarm branch fm/seizure-crosssite (85cb3a2).
# Every run writes a .done marker on /shared and is skipped on a rerun; a data fetch skips files it already has.
set -euo pipefail
unset CDPATH
REPO=$PWD
ROOT=${FILL_ROOT:-/shared/aayu}
D=$ROOT/fill-cpu
mkdir -p "$D"/{data,runs,logs,bin} "$ROOT/bin"
log() { printf '== %s %s\n' "$*" "$(date -u +%FT%TZ)"; }

# The repo's fetch scripts call shasum (Perl, not on every node) and python3, and aws for SeizeIT2.
VENV=$ROOT/venv-cpu
exec 9>"$ROOT/venv-cpu.lock"; flock 9
if [[ ! -f $VENV/.ready ]]; then
  [[ -x $ROOT/bin/uv ]] || curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$ROOT/bin" UV_NO_MODIFY_PATH=1 sh
  export UV_PYTHON_INSTALL_DIR=$ROOT/python UV_CACHE_DIR=$ROOT/uv-cache
  rm -rf "$VENV"; "$ROOT/bin/uv" venv -q --python 3.12 "$VENV"
  "$ROOT/bin/uv" pip install -q --python "$VENV/bin/python" numpy scipy "scikit-learn>=1.5,<1.9" "mne>=1.6,<2" awscli
  touch "$VENV/.ready"
fi
flock -u 9
printf '#!/usr/bin/env bash\nif [[ $1 == -a ]]; then shift 2; fi\nexec sha256sum "$@"\n' > "$D/bin/shasum"; chmod +x "$D/bin/shasum"
export PATH=$D/bin:$VENV/bin:$PATH PYTHONPATH=$REPO/src PYTHONUNBUFFERED=1
"$VENV/bin/python" -c 'import sklearn, mne, seizure_alarm; print("env ok", sklearn.__version__, mne.__version__)'

# ---- data ----
export CHBMIT_ROOT=$D/data/chbmit SIENA_ROOT=$D/data/siena SEIZEIT2_ROOT=$D/data/seizeit2
fetch() {  # fetch <name> <script>: skipped once its .done marker exists
  [[ -f $D/data/$1.done ]] && { log "skip fetch $1"; return 0; }
  log "fetch $1"; JOBS=16 bash "$REPO/scripts/$2" && touch "$D/data/$1.done"; log "fetched $1"
}
fetch chbmit fetch_chbmit.sh
fetch siena fetch_siena.sh
fetch seizeit2 fetch_seizeit2.sh > "$D/logs/fetch_seizeit2.log" 2>&1 &
sz_pid=$!

# ---- runs: 64 CPUs shared by the runs, each capped by its own thread limit ----
run() {  # run <name> <threads> <benchmark args...>
  local name=$1 thr=$2; shift 2
  [[ -f $D/runs/$name/.done ]] && { log "skip run $name"; return 0; }
  log "start run $name"
  OMP_NUM_THREADS=$thr OPENBLAS_NUM_THREADS=$thr MKL_NUM_THREADS=$thr "$VENV/bin/python" "$REPO/scripts/benchmark.py" "$@" --out "$D/runs/$name" \
    > "$D/logs/$name.log" 2>&1 || { tail -30 "$D/logs/$name.log"; return 1; }
  touch "$D/runs/$name/.done"; log "done run $name"
}
pids=()
run chbmit 12 --dataset chbmit --root "$CHBMIT_ROOT" & pids+=($!)
run siena 8 --dataset siena --root "$SIENA_ROOT" & pids+=($!)
run chbmit-to-siena 8 --dataset siena --root "$SIENA_ROOT" --source chbmit --source-root "$CHBMIT_ROOT" & pids+=($!)
wait "$sz_pid"  # SeizeIT2 is about 126 GB: its two runs start when it has landed
run seizeit2 24 --dataset seizeit2 --root "$SEIZEIT2_ROOT" & pids+=($!)
run chbmit-to-seizeit2 12 --dataset seizeit2 --root "$SEIZEIT2_ROOT" --source chbmit --source-root "$CHBMIT_ROOT" & pids+=($!)
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done

for r in chbmit siena chbmit-to-siena seizeit2 chbmit-to-seizeit2; do
  echo "FILL_SUMMARY_BEGIN $r"; cat "$D/runs/$r/summary.json" 2>/dev/null || echo "(no summary: run failed)"; echo; echo "FILL_SUMMARY_END $r"
done
((rc == 0)) || { echo "a run failed" >&2; exit 1; }
echo "FILL_CPU_DONE"
