"""Arousal census for NSRR annotation XMLs (issue #83 pre-registration input).

NSRR mostly does not label arousal subtype, so arousals are classified by timing:
  respiratory: onset within [resp_start, resp_end + RESP_GAP_S] of an apnea/hypopnea
  limb:        onset within LIMB_GAP_S of a limb/PLM movement (and not respiratory)
  spontaneous: everything else
"clean" = spontaneous, onset in sleep, and no arousal/wake/respiratory/limb event in the
preceding CLEAN_PRE_S seconds, i.e. a usable strictly-future forecasting target.

Usage: python scripts/nsrr_arousal_census.py ~/data/nsrr/mesa ~/data/nsrr/shhs --out artifacts/nsrr_arousal_census
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

RESP_GAP_S = 5.0  # ponytail: fixed AASM-style window. Sensitivity sweep (3/5/10 s) is the upgrade path.
LIMB_GAP_S = 0.5
CLEAN_PRE_S = 60.0
SLEEP_STAGES = {"Stage 1 sleep", "Stage 2 sleep", "Stage 3 sleep", "Stage 4 sleep", "REM sleep"}


def parse(xml_path: Path) -> list[tuple[str, float, float]]:
    out = []
    for ev in ET.parse(xml_path).getroot().iter("ScoredEvent"):
        concept = (ev.findtext("EventConcept") or "").split("|")[0].strip()
        try:
            out.append((concept, float(ev.findtext("Start")), float(ev.findtext("Duration"))))
        except (TypeError, ValueError):
            continue
    return out


def classify(events: list[tuple[str, float, float]]) -> dict:
    low = [(c.lower(), s, d) for c, s, d in events]
    arousals = sorted(s for c, s, d in low if "arousal" in c)
    resp = [(s, s + d) for c, s, d in low if "apnea" in c or "hypopnea" in c]
    limb = [(s, s + d) for c, s, d in low if "limb" in c or "leg movement" in c]
    stages = sorted((s, s + d, c) for c, s, d in events if c in SLEEP_STAGES or c == "Wake")

    def stage_at(t: float) -> str | None:
        for s, e, c in stages:
            if s <= t < e:
                return c
        return None

    counts = {"respiratory": 0, "limb": 0, "spontaneous": 0, "clean_spontaneous": 0}
    disturb = sorted([s for s in arousals] + [s for s, _ in resp] + [s for s, _ in limb]
                     + [s for s, e, c in stages if c == "Wake"])
    for t in arousals:
        if any(s <= t <= e + RESP_GAP_S for s, e in resp):
            counts["respiratory"] += 1
            continue
        if any(s - LIMB_GAP_S <= t <= e + LIMB_GAP_S for s, e in limb):
            counts["limb"] += 1
            continue
        counts["spontaneous"] += 1
        stage = stage_at(t)
        prior = [x for x in disturb if t - CLEAN_PRE_S <= x < t]
        if stage in SLEEP_STAGES and not prior and stage_at(t - CLEAN_PRE_S) in SLEEP_STAGES:
            counts["clean_spontaneous"] += 1
    counts["total"] = len(arousals)
    tst_s = sum(e - s for s, e, c in stages if c in SLEEP_STAGES)
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
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("artifacts/nsrr_arousal_census"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    report = {}
    for d in a.dirs:
        rows = []
        for x in sorted(d.glob("*-nsrr.xml")):
            try:
                rows.append({"subject": x.stem.replace("-nsrr", ""), **classify(parse(x))})
            except ET.ParseError:
                print(f"skip unparseable {x}", file=sys.stderr)
        report[d.name] = summarize(rows)
        with open(a.out / f"{d.name}_per_subject.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    (a.out / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


def _self_check() -> None:
    ev = [("Stage 2 sleep", 0, 600), ("Hypopnea", 100, 20), ("Arousal", 122, 5),
          ("Limb movement - left", 300, 2), ("Arousal", 300.2, 4), ("Arousal", 450, 5), ("Arousal", 470, 5)]
    c = classify(ev)
    assert (c["respiratory"], c["limb"], c["spontaneous"], c["clean_spontaneous"]) == (1, 1, 2, 1), c


if __name__ == "__main__":
    _self_check()
    raise SystemExit(main())
