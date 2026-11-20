#!/usr/bin/env bash
# Mirror first (CPU only), then the 4-GPU pretraining stage. Run from the repo root, on the launching server.
#   infra/engram/fill/chain.sh <log-dir>
# Each stage relaunches after a stop until its done marker appears in its log; the clusters go down at the end.
set -uo pipefail
D=${1:?log dir}
SKY=~/.venvs/sky/bin/sky
export NSRR_TOKEN=$(cat ~/.nsrr_token)
until grep -q NSRR_MIRROR_DONE "$D/fill-nsrr.log" 2>/dev/null; do
  pgrep -u "$(id -u)" -f "sky launch -c fill-nsrr" > /dev/null || \
    $SKY launch -c fill-nsrr --down -y --secret NSRR_TOKEN infra/engram/fill/fill_nsrr.yaml >> "$D/fill-nsrr.log" 2>&1
  sleep 60
done
grep -E "NSRR_MIRROR_(COUNTS|DONE)" "$D/fill-nsrr.log" | tail -2 > "$D/mirror-done.txt"
exec infra/engram/fill/run_until_done.sh "$D/fill-a.log"
