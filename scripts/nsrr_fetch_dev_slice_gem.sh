#!/usr/bin/env bash
# Fetch only the 50 dev-slice nights (25 MESA + 25 SHHS-1: EDF + scored-event XML) with the NSRR gem (Ruby).
# The token comes from NSRR_TOKEN (default: contents of ~/.nsrr_token) and never goes on a command line.
# SSL_CERT_FILE must name a CA bundle with the Sectigo DV R36 intermediate (sleepdata.org sends an incomplete chain).
# Files land in the gem's folder layout under NSRR_ROOT (default ~/data/nsrr). Point the studies at <cohort>/polysomnography (split layout).
# Usage: scripts/nsrr_fetch_dev_slice_gem.sh
set -euo pipefail
repo=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
export NSRR_TOKEN=${NSRR_TOKEN:-$(cat ~/.nsrr_token)}
export SSL_CERT_FILE=${SSL_CERT_FILE:-$HOME/.config/nsrr/ca-bundle.pem}
root=${NSRR_ROOT:-$HOME/data/nsrr}
umask 077
mkdir -p "$root"; cd "$root"
prov=$repo/docs/results/provenance
ids() { sed -E 's#.*/(.*)\.edf#\1#' "$prov/$1_dev_slice_edfs.txt" | paste -sd'|'; }
mesa=$(ids mesa); shhs=$(ids shhs)
ruby "$repo/scripts/nsrr_fetch.rb" mesa/polysomnography/edfs "^(${mesa})\.edf$"
ruby "$repo/scripts/nsrr_fetch.rb" mesa/polysomnography/annotations-events-nsrr "^(${mesa})-nsrr\.xml$"
ruby "$repo/scripts/nsrr_fetch.rb" shhs/polysomnography/edfs/shhs1 "^(${shhs})\.edf$"
ruby "$repo/scripts/nsrr_fetch.rb" shhs/polysomnography/annotations-events-nsrr/shhs1 "^(${shhs})-nsrr\.xml$"
du -sh "$root"/*
