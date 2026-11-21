#!/usr/bin/env bash
# Launches one E1 task and relaunches it after any stop until its done marker is in the log. Run from the repo root.
#   infra/engram/e1/until_done.sh <cluster> <yaml> <done marker> <log file> [extra sky launch args]
set -uo pipefail
C=$1 Y=$2 M=$3 LOG=$4; shift 4
until grep -q "$M" "$LOG" 2>/dev/null; do
  ~/.venvs/sky/bin/sky launch -c "$C" --down -i 1 -y "$@" "$Y" >> "$LOG" 2>&1
  grep -q "$M" "$LOG" || sleep 120
done
