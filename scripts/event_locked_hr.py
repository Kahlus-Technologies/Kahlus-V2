"""Event-locked heart rate around clean spontaneous arousals (issue #83).

Answers: how early before arousal onset does heart rate start to move? That bounds
how far ahead any heart-only forecaster can see.

HR is linearly interpolated between beats, so the value at second s can include
the next beat after s (up to ~1 s of look-ahead). Read the last second before
onset with that in mind.

Usage: python scripts/event_locked_hr.py ~/data/nsrr/mesa:EKG ~/data/nsrr/shhs:ECG --out artifacts/event_locked_hr
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from neurotwin.adapters.nsrr import classify_arousals, discover_nsrr_recordings, load_nsrr_epoch_matrix, parse_nsrr_events
from neurotwin.adapters.qrs_fast import beat_intervals, detect_qrs

PRE_S, POST_S, BASELINE_S = 60, 15, 30
MAX_MEDIAN_BPM = 110  # nights above this are QRS double-counting, not physiology


def cohort_curves(root: Path, ecg_label: str) -> tuple[np.ndarray, int]:
    curves, dropped = [], 0
    for rec in discover_nsrr_recordings(root, dataset=root.name):
        m = load_nsrr_epoch_matrix(rec.edf_path, channel_groups={"ecg": (ecg_label,)})
        fs = m["sfreq"]
        bt, rr = beat_intervals(detect_qrs(m["epochs"]["ecg"][:, :, 0].ravel(), fs), fs)
        if rr.size < 100 or np.median(60 / rr) > MAX_MEDIAN_BPM:
            dropped += 1
            continue
        hr = np.interp(np.arange(0, bt[-1]), bt, 60 / rr)
        for a in classify_arousals(parse_nsrr_events(rec.xml_path)):
            o = int(a.onset)
            if a.clean and o > PRE_S + 10 and o + POST_S < hr.size:
                seg = hr[o - PRE_S : o + POST_S]
                curves.append(seg - seg[:BASELINE_S].mean())
    return np.asarray(curves), dropped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cohorts", nargs="+", help="root:ecg_label")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    report = {}
    for spec in a.cohorts:
        root, label = spec.rsplit(":", 1)
        c, dropped = cohort_curves(Path(root).expanduser(), label)
        mu, ci = c.mean(0), 1.96 * c.std(0) / np.sqrt(len(c))
        t = np.arange(-PRE_S, POST_S)
        # First second (scanning back from onset) where the 95% CI still excludes 0 on the rise side.
        rising = (mu - ci) > 0
        lead = 0
        for s in range(PRE_S - 1, -1, -1):
            if not rising[s]:
                break
            lead = PRE_S - s
        report[Path(root).name] = {
            "n_arousals": int(len(c)),
            "nights_dropped_bad_ecg": dropped,
            "earliest_significant_rise_s_before_onset": int(lead),
            "delta_hr_bpm": {int(s): round(float(mu[i]), 3) for i, s in enumerate(t)},
            "ci95_bpm": {int(s): round(float(ci[i]), 3) for i, s in enumerate(t)},
        }
        print(Path(root).name, len(c), "arousals; HR significantly up from", lead, "s before onset;",
              " ".join(f"{s:+d}s:{mu[PRE_S + s]:+.2f}" for s in (-10, -5, -3, -2, -1, 0, 2, 5)))
    (a.out / "report.json").write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
