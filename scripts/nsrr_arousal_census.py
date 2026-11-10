"""Arousal census for NSRR annotation XMLs (issue #83 pre-registration input).

NSRR mostly does not label arousal subtype, so arousals are classified by timing:
  respiratory: onset within [resp_start, resp_end + RESP_GAP_S] of an apnea/hypopnea
  limb:        onset within LIMB_GAP_S of a limb/PLM movement (and not respiratory)
  spontaneous: everything else
"clean" = spontaneous, onset in sleep, and no arousal/wake/respiratory/limb event in the
preceding CLEAN_PRE_S seconds, i.e. a usable strictly-future forecasting target.

Usage: python scripts/nsrr_arousal_census.py ~/data/nsrr/mesa ~/data/nsrr/shhs --out artifacts/nsrr_arousal_census

Each dir may be co-located or split (edfs/ + annotations/). Without dirs it reads
nsrr_root("mesa") and nsrr_root("shhs"). NSRR_<COHORT>_IDS limits the participants.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET  # ParseError
from pathlib import Path

import numpy as np

from kahlus.nsrr import (
    CLEAN_PRE_S,
    LIMB_GAP_S,
    RESP_GAP_S,
    SLEEP_STAGES,
    classify_arousals,
    discover_nsrr_annotations,
    nsrr_root,
    parse_nsrr_events,
    sleep_stage_epochs,
)


def classify(events: list[tuple[str, float, float]]) -> dict:
    arousals = classify_arousals(events)
    counts = {k: sum(a.kind == k for a in arousals) for k in ("respiratory", "limb", "spontaneous")}
    counts["clean_spontaneous"] = sum(a.clean for a in arousals)
    counts["total"] = len(arousals)
    tst_s = sum(e - s for s, e, c in sleep_stage_epochs(events) if c in SLEEP_STAGES)
    counts["tst_h"] = round(tst_s / 3600, 3)
    return counts


def summarize(rows: list[dict]) -> dict:
    keys = ["total", "respiratory", "limb", "spontaneous", "clean_spontaneous"]
    s = {k: int(sum(r[k] for r in rows)) for k in keys}
    clean = np.array([r["clean_spontaneous"] for r in rows])
    s.update(
        n_subjects=len(rows),
        frac_spontaneous=round(s["spontaneous"] / max(s["total"], 1), 3),
        clean_per_subject_median=float(np.median(clean)) if len(clean) else 0.0,
        clean_per_subject_iqr=[float(x) for x in np.percentile(clean, [25, 75])] if len(clean) else [0, 0],
        subjects_with_ge10_clean=int((clean >= 10).sum()),
        params={"RESP_GAP_S": RESP_GAP_S, "LIMB_GAP_S": LIMB_GAP_S, "CLEAN_PRE_S": CLEAN_PRE_S},
    )
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", type=Path)
    ap.add_argument("--out", type=Path, default=Path("artifacts/nsrr_arousal_census"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    report = {}
    for d in a.dirs or [nsrr_root("mesa"), nsrr_root("shhs")]:
        rows = []
        xmls, found = discover_nsrr_annotations(d, dataset=d.name)  # MESA seal applies to annotations too
        name = found.cohort or d.name
        unparseable = 0
        for _, x in xmls:
            try:
                rows.append({"subject": x.name.removesuffix(".xml").removesuffix("-nsrr"), **classify(parse_nsrr_events(x))})
            except ET.ParseError:
                unparseable += 1
                print(f"skip unparseable {x}", file=sys.stderr)
        report[name] = summarize(rows) | {"unparseable_xml": unparseable, "discovery": found.to_json()}
        print(found.summary(), file=sys.stderr)
        with open(a.out / f"{name}_per_subject.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    (a.out / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
