#!/usr/bin/env bash
# Prints the path of a Python that has torch and braindecode, building it once on /shared.
# Jobs on Engram start from a bare Python 3.14 with no torch (measured 2026-11-19). The first job
# builds a venv keyed by the pinned requirements; every later job finds `.ready` and skips the install.
# A lock stops two jobs that start together from building the same venv twice.
#
#   PY=$(bash setup_env.sh)        # in a SkyPilot `run` block
set -euo pipefail
ROOT=${KAHLUS_ROOT:-/shared/$USER/kahlus}
# torchaudio must match torch: braindecode imports it, and an unpinned one is built for another CUDA.
REQ="torch==2.9.1 torchaudio==2.9.1 braindecode==2.0.0 numpy==2.5.5"
PYVER=3.12  # braindecode 2.0 needs 3.11 or newer. 3.12 avoids the brand-new 3.14 of the host
VENV=$ROOT/venv-$(printf '%s %s' "$REQ" "$PYVER" | sha256sum | cut -c1-12)
mkdir -p "$ROOT/bin"
exec 9>"$ROOT/venv.lock"
flock 9
if [[ ! -f $VENV/.ready ]]; then
  {
    [[ -x $ROOT/bin/uv ]] || curl -LsSf https://astral.sh/uv/0.13.9/install.sh | env UV_INSTALL_DIR="$ROOT/bin" UV_NO_MODIFY_PATH=1 sh
    rm -rf "$VENV"
    export UV_PYTHON_INSTALL_DIR=$ROOT/python UV_CACHE_DIR=$ROOT/uv-cache
    "$ROOT/bin/uv" venv -q --python "$PYVER" "$VENV"
    "$ROOT/bin/uv" pip install -q --python "$VENV/bin/python" $REQ
    "$VENV/bin/python" -c 'import torch, braindecode.models; print("env ok", torch.__version__, torch.version.cuda)'
    touch "$VENV/.ready"  # written last: a half-built venv is never marked ready
  } >&2
fi
echo "$VENV/bin/python"
