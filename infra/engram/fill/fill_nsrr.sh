#!/usr/bin/env bash
# The single NSRR mirror on Engram: SHHS-1 annotations and EDFs into /shared/aayu/nsrr, CPU only, 20 parallel
# streams with retries. Resumable: the gem skips files whose size already matches, so a rerun fetches the rest.
# Layout (what kahlus.nsrr reads with NSRR_SHHS_ROOT=/shared/aayu/nsrr/shhs/polysomnography):
#   /shared/aayu/nsrr/shhs/polysomnography/edfs/shhs1/shhs1-<id>.edf
#   /shared/aayu/nsrr/shhs/polysomnography/annotations-events-nsrr/shhs1/shhs1-<id>-nsrr.xml
set -euo pipefail
unset CDPATH
REPO=$PWD
ROOT=${FILL_ROOT:-/shared/aayu}
NSRR=$ROOT/nsrr
mkdir -p "$NSRR" "$ROOT/fill-nsrr/logs"
umask 077; chmod 700 "$ROOT" "$NSRR"
export SSL_CERT_FILE=$HOME/.nsrr-ca.pem  # system CAs plus the Sectigo DV R36 intermediate
RUBY=$(bash "$REPO/infra/engram/fill/setup_ruby.sh")
LOGS=$ROOT/fill-nsrr/logs
fetch() {  # fetch <path> [regex]: up to 5 attempts; the gem resumes by skipping complete files
  local i
  for i in 1 2 3 4 5; do
    ( cd "$NSRR" && "$RUBY" "$REPO/infra/engram/fill/nsrr_fetch.rb" "$@" ) && return 0
    echo "retry $i: $*" >&2; sleep $((i * 20))
  done
  return 1
}
echo "== xml $(date -u +%FT%TZ)"
fetch shhs/polysomnography/annotations-events-nsrr/shhs1 2>&1 | tail -n 15
echo "== edf, 20 streams $(date -u +%FT%TZ)"
pids=()
for t in '[0-4]' '[5-9]'; do   # 20 streams: tens digit half x units digit of the 6-digit ID
  for d in 0 1 2 3 4 5 6 7 8 9; do
    fetch shhs/polysomnography/edfs/shhs1 "^shhs1-[0-9]{4}${t}${d}\.edf\$" > "$LOGS/edf-$t$d.log" 2>&1 &
    pids+=($!)
  done
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
((rc == 0)) || tail -n 3 "$LOGS"/edf-*.log
E=$NSRR/shhs/polysomnography/edfs/shhs1 X=$NSRR/shhs/polysomnography/annotations-events-nsrr/shhs1
bad=0; for f in "$E"/*.edf; do [[ $(head -c 8 "$f") == "0       " ]] || { echo "bad EDF $f"; bad=$((bad + 1)); }; done
echo "NSRR_MIRROR_COUNTS edf=$(ls "$E"/*.edf | wc -l) xml=$(ls "$X"/*.xml | wc -l) bad_edf=$bad stream_failures=$rc"
du -sh "$NSRR/shhs"; stat -c '%a %n' "$NSRR"
((rc == 0 && bad == 0)) || exit 1
echo "NSRR_MIRROR_DONE root=$NSRR"
