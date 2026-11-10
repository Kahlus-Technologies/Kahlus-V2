"""Write what NSRR discovery finds for one cohort, and optionally a fixed subset of it.

Writes OUT/discovery_<cohort>.json (layout, counts of matched, unmatched and
duplicate files, ID-filter and MESA-seal counts). With --limit N it also writes
OUT/ids_<cohort>.txt: the N participants that come first when ordered by
sha256("<seed>:<subject id>"). The same data, seed and N always give the same
subset, and adding nights to the root only adds candidates; it doesn't reshuffle
the rest. Point NSRR_<COHORT>_IDS at that file to make every study use the subset.

Exits nonzero when no recording is usable, so a wrong root stops a run early.

Usage: python scripts/nsrr_discover.py --cohort shhs --root /data/shhs --out RUN/discovery --limit 50 --seed 0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from neurotwin.adapters.nsrr import NSRR_COHORTS, discover_nsrr, ids_sha256, nsrr_root


def subset(ids: list[str], limit: int, seed: int) -> list[str]:
    ranked = sorted(ids, key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).hexdigest())
    return sorted(ranked[:limit])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", choices=NSRR_COHORTS, required=True)
    ap.add_argument("--root", type=Path, default=None, help="default: nsrr_root(cohort)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.limit is not None and a.limit < 1:
        ap.error("--limit must be at least 1")
    recs, found = discover_nsrr(a.root or nsrr_root(a.cohort), dataset=a.cohort)
    a.out.mkdir(parents=True, exist_ok=True)
    body = found.to_json()
    if a.limit is not None:
        ids = subset([r.subject_id for r in recs], a.limit, a.seed)
        path = a.out / f"ids_{a.cohort}.txt"
        with open(path, "x") as f:  # never overwrite a subset a run already used
            f.write(f"# {len(ids)} of {len(recs)} {a.cohort} nights, first by sha256('{a.seed}:<id>')\n")
            f.writelines(i + "\n" for i in ids)
        body["subset"] = {"file": str(path.resolve()), "limit": a.limit, "seed": a.seed, "n": len(ids),
                          "ids_sha256": ids_sha256(ids)}
    with open(a.out / f"discovery_{a.cohort}.json", "x") as f:
        json.dump(body, f, indent=2)
    if not recs:
        print(f"no usable {a.cohort} recordings under {found.root}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
