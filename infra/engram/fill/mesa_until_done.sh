#!/usr/bin/env bash
# Runs fill_mesa.yaml and relaunches it after a stall (exit 75) until NSRR_MIRROR_DONE is in the log. CPU only; run from the repo root.
#   infra/engram/fill/mesa_until_done.sh <log-file>
# Never run it beside the SHHS mirror: sleepdata.org allows about 10 concurrent requests in total.
set -uo pipefail
LOG=${1:?log file}
export NSRR_TOKEN=$(cat ~/.nsrr_token)
until grep -q NSRR_MIRROR_DONE "$LOG" 2>/dev/null; do
  ~/.venvs/sky/bin/sky launch -c fill-mesa --down -y --secret NSRR_TOKEN infra/engram/fill/fill_mesa.yaml >> "$LOG" 2>&1
  grep -q NSRR_MIRROR_DONE "$LOG" || sleep 60
done
