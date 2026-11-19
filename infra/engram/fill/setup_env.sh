#!/usr/bin/env bash
# Prints the path of a Python with CUDA torch and the kahlus dependencies, built once into /shared/aayu/venv.
# Engram jobs run without root and start from a bare Python with no torch, so the first job builds the venv
# and every later job finds `.ready` and skips the install. The lock stops two jobs from building it twice.
set -euo pipefail
ROOT=${FILL_ROOT:-/shared/aayu}
VENV=$ROOT/venv
REQ="torch==2.9.1 numpy scipy scikit-learn>=1.5,<1.9 mne>=1.6,<2"  # kahlus[train] without an editable install: jobs set PYTHONPATH
mkdir -p "$ROOT/bin"
exec 9>"$ROOT/venv.lock"
flock 9
if [[ ! -f $VENV/.ready || $(cat "$VENV/.req" 2>/dev/null) != "$REQ" ]]; then
  {
    [[ -x $ROOT/bin/uv ]] || curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$ROOT/bin" UV_NO_MODIFY_PATH=1 sh
    export UV_PYTHON_INSTALL_DIR=$ROOT/python UV_CACHE_DIR=$ROOT/uv-cache
    rm -rf "$VENV"
    "$ROOT/bin/uv" venv -q --python 3.12 "$VENV"
    # shellcheck disable=SC2086  # REQ is a list of requirement strings on purpose
    "$ROOT/bin/uv" pip install -q --python "$VENV/bin/python" "torch==2.9.1" "numpy" "scipy" "scikit-learn>=1.5,<1.9" "mne>=1.6,<2"
    "$VENV/bin/python" -c 'import torch; print("env ok", torch.__version__, torch.version.cuda)'
    printf %s "$REQ" > "$VENV/.req"
    touch "$VENV/.ready"  # written last: a half-built venv is never marked ready
  } >&2
fi
echo "$VENV/bin/python"
