#!/usr/bin/env bash
# Launches fill_a.yaml and relaunches it after any stop until the job prints FILL_A_DONE, then takes the cluster down.
# Run from the repo root. The token comes from ~/.nsrr_token into a SkyPilot secret; the job log keeps only the summaries.
set -uo pipefail
LOG=${1:-fill-a.log}
export NSRR_TOKEN=$(cat ~/.nsrr_token)
until grep -q '^.*FILL_A_DONE' "$LOG" 2>/dev/null; do
  ~/.venvs/sky/bin/sky launch -c fill-a --down -y --secret NSRR_TOKEN --env FILL_COMMIT="$(git rev-parse HEAD)" infra/engram/fill/fill_a.yaml 2>&1 | tee -a "$LOG"
  sleep 60
done
~/.venvs/sky/bin/sky down -y fill-a
