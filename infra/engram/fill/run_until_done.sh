#!/usr/bin/env bash
# Launches fill_a.yaml and relaunches it after any stop until the job prints FILL_A_DONE, then takes the cluster down.
# Run from the repo root. The token comes from ~/.nsrr_token into a SkyPilot secret; the job log keeps only the summaries.
#   infra/engram/fill/run_until_done.sh fill-a.log                       # 4 GPUs: download, shards, 10M, 50M
#   GPUS=8 SIZES=200m REQUIRE_SHARDS=1 infra/engram/fill/run_until_done.sh fill-a200.log   # 200M burst, then drop back
set -uo pipefail
LOG=${1:-fill-a.log} GPUS=${GPUS:-4} SIZES=${SIZES:-"10m 50m"} REQUIRE_SHARDS=${REQUIRE_SHARDS:-0}
export NSRR_TOKEN=$(cat ~/.nsrr_token)
until grep -q 'FILL_A_DONE' "$LOG" 2>/dev/null; do
  ~/.venvs/sky/bin/sky launch -c fill-a --down -y --gpus "H100:$GPUS" --secret NSRR_TOKEN --env FILL_COMMIT="$(git rev-parse HEAD)" \
    --env NGPU="$GPUS" --env SIZES="$SIZES" --env REQUIRE_SHARDS="$REQUIRE_SHARDS" infra/engram/fill/fill_a.yaml 2>&1 | tee -a "$LOG"
  grep -q 'FILL_A_DONE' "$LOG" 2>/dev/null || sleep 60
done
~/.venvs/sky/bin/sky down -y fill-a
