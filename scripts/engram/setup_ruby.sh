#!/usr/bin/env bash
# Prints the path of a Ruby that has the nsrr gem, building it once in the user's /shared folder.
# Engram jobs run without root, so Ruby comes from conda-forge through micromamba, not apt.
#
#   RUBY=$(bash setup_ruby.sh)
set -euo pipefail
ROOT=${KAHLUS_ROOT:-/shared/$USER/kahlus}
ENV_DIR=$ROOT/ruby-nsrr-8.0.0
mkdir -p "$ROOT/bin"
exec 9>"$ROOT/ruby.lock"
flock 9
if [[ ! -f $ENV_DIR/.ready ]]; then
  {
    [[ -x $ROOT/bin/micromamba ]] || curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$ROOT" bin/micromamba
    rm -rf "$ENV_DIR"
    MAMBA_ROOT_PREFIX=$ROOT/mamba "$ROOT/bin/micromamba" create -y -q -p "$ENV_DIR" -c conda-forge "ruby>=3.2"
    GEM_HOME=$ENV_DIR/gems "$ENV_DIR/bin/gem" install --no-document nsrr -v 8.0.0
    touch "$ENV_DIR/.ready"  # written last: a half-built env is never marked ready
  } >&2
fi
# A wrapper, so callers need no GEM_HOME of their own.
printf '#!/usr/bin/env bash\nGEM_HOME=%q exec %q "$@"\n' "$ENV_DIR/gems" "$ENV_DIR/bin/ruby" > "$ENV_DIR/ruby-nsrr"
chmod +x "$ENV_DIR/ruby-nsrr"
echo "$ENV_DIR/ruby-nsrr"
