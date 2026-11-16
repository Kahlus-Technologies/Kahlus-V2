#!/usr/bin/env bash
# Fetch NSRR data: all MESA+SHHS1 annotation XMLs plus the first N_EDF EDFs of each cohort (default 25: the dev slice).
# N_SHHS_EDF overrides the SHHS count; 0 means every SHHS-1 EDF.
# Idempotent: skips files already present. Token read from ~/.nsrr_token. Writes to NSRR_ROOT (default ~/data/nsrr).
set -euo pipefail
T=$(cat ~/.nsrr_token)
ROOT=${NSRR_ROOT:-$HOME/data/nsrr}
N_EDF=${N_EDF:-25}
api() { curl -s "https://sleepdata.org/api/v1/datasets/$1/files.json?auth_token=$T&path=$2"; }
url() { echo "https://sleepdata.org/datasets/$1/files/a/$T/m/nsrr-gem-v8-0-0/$2"; }
list() { api "$1" "$2" | python3 -c 'import json,sys;[print(x["full_path"]) for x in json.load(sys.stdin) if x.get("is_file")]'; }
get() { # dataset path -> ROOT/dataset/basename
  local out="$ROOT/$1/$(basename "$2")"
  [ -s "$out" ] && return 0
  if curl -sfL --retry 3 -o "$out.part" "$(url "$1" "$2")"; then mv -f "$out.part" "$out"; else echo "FAIL $2" >&2; fi
}
export -f get url; export T ROOT

for spec in "mesa $N_EDF polysomnography/annotations-events-nsrr polysomnography/edfs" \
            "shhs ${N_SHHS_EDF:-$N_EDF} polysomnography/annotations-events-nsrr/shhs1 polysomnography/edfs/shhs1"; do
  set -- $spec; ds=$1; mkdir -p "$ROOT/$ds"
  list "$ds" "$3" > "$ROOT/$ds/xml_list.txt"
  list "$ds" "$4" | awk -v n="$2" 'n == 0 || NR <= n' > "$ROOT/$ds/edf_list.txt"
  echo "$ds: $(wc -l < "$ROOT/$ds/xml_list.txt") xml, $(wc -l < "$ROOT/$ds/edf_list.txt") edf"
  xargs -P 8 -I{} bash -c "get $ds {}" < "$ROOT/$ds/xml_list.txt"
  xargs -P 4 -I{} bash -c "get $ds {}" < "$ROOT/$ds/edf_list.txt"
done
du -sh "$ROOT"/*
