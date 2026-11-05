"""Low-cost montage stress test for arousal detection (issue #97).

A single-channel C4 arousal *detector* (not forecaster) is trained on clean PSG
from one cohort and tested on the other cohort's EEG after headband-like
degradation (montage_degrade.LEVELS). Event-level scoring: detected onsets are
matched one-to-one to scored arousal onsets within TOL_S.

Per second t during scored sleep: features from the 3 s of EEG ending at t
(eeg_features) plus the same features causally z-scored within the night. Label:
t lies inside any scored arousal. Threshold picked on the training cohort to
maximize event F1, then frozen.

Usage: python scripts/run_montage_stress.py --mesa ~/data/nsrr/mesa --shhs ~/data/nsrr/shhs --out artifacts/montage_stress_dev25
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


from neurotwin.adapters.nsrr import (
    SLEEP_STAGES,
    discover_nsrr_recordings,
    event_kind,
    load_nsrr_epoch_matrix,
    parse_nsrr_events,
    sleep_stage_epochs,
)
from neurotwin.forecastability.heart_arousal import COHORTS, causal_z, eeg_features, make_model
from neurotwin.forecastability.montage_degrade import LEVELS, degrade, detected_onsets, match_events

TOL_S = 3.0
WINDOW_S = 3.0


def night_rows(eeg_uv: np.ndarray, fs: float, events, level: str, seed: int) -> dict:
    x = degrade(eeg_uv, fs, level, seed=seed)
    n_sec = int(x.size / fs)
    asleep = np.zeros(n_sec, dtype=bool)
    for s, e, c in sleep_stage_epochs(events):
        if c in SLEEP_STAGES:
            asleep[int(s) : min(int(e), n_sec)] = True
    inside = np.zeros(n_sec, dtype=bool)
    onsets = []
    for c, s, d in events:
        if event_kind(c) == "arousal":
            inside[int(s) : min(int(np.ceil(s + d)), n_sec)] = True  # second t is labeled if it starts inside [s, s+d)
            if int(s) < n_sec and asleep[int(s)]:
                onsets.append(s)
    t = np.arange(int(WINDOW_S), n_sec, dtype=np.float64)
    f = eeg_features(x, fs, t, window_s=WINDOW_S)
    feats = np.concatenate([f, causal_z(f)], axis=1)
    keep = asleep[t.astype(int)]
    return {"t": t[keep], "x": feats[keep], "y": inside[t.astype(int)][keep].astype(int), "onsets": np.asarray(onsets)}


def load_eeg(cohort: str, root: Path) -> list[tuple[np.ndarray, float, list]]:
    out = []
    for rec in discover_nsrr_recordings(root, dataset=cohort):
        m = load_nsrr_epoch_matrix(rec.edf_path, channel_groups={"eeg": COHORTS[cohort]["eeg_labels"]})
        out.append((m["epochs"]["eeg"][:, :, 0].ravel() * 1e6, m["sfreq"], parse_nsrr_events(rec.xml_path)))
    return out


def score_nights(nights: list[dict], prob: list[np.ndarray], thr: float) -> dict:
    # Offset each night far apart on one time axis so matches never cross nights.
    off, d_all, t_all = 0.0, [], []
    for n, p in zip(nights, prob):
        d_all.append(detected_onsets(p, n["t"], thr) + off)
        t_all.append(n["onsets"] + off)
        off += (n["t"][-1] if n["t"].size else 0.0) + 1e5
    m = match_events(np.concatenate(d_all), np.concatenate(t_all), TOL_S)
    hours = sum(n["t"].size for n in nights) / 3600.0
    m["false_detections_per_hour"] = (m["n_detected"] - m["tp"]) / hours
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesa", type=Path, required=True)
    ap.add_argument("--shhs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    raw = {c: load_eeg(c, getattr(a, c)) for c in ("mesa", "shhs")}
    clean = {c: [night_rows(e, fs, ev, "clean", i) for i, (e, fs, ev) in enumerate(raw[c])] for c in raw}
    report = {"tol_s": TOL_S, "levels": list(LEVELS), "results": {}}
    for src, dst in (("mesa", "shhs"), ("shhs", "mesa")):
        tr = clean[src]
        x, y = np.concatenate([n["x"] for n in tr]), np.concatenate([n["y"] for n in tr])
        ok = ~np.isnan(x).all(axis=1)
        clf = make_model(0).fit(x[ok], y[ok])
        p_tr = [clf.predict_proba(n["x"])[:, 1] for n in tr]
        thr = max(np.linspace(0.05, 0.9, 18), key=lambda th: score_nights(tr, p_tr, th)["f1"])
        rows = {"threshold": float(thr), "train_f1": score_nights(tr, p_tr, thr)["f1"]}
        for level in LEVELS:
            te = clean[dst] if level == "clean" else [night_rows(e, fs, ev, level, i) for i, (e, fs, ev) in enumerate(raw[dst])]
            rows[level] = score_nights(te, [clf.predict_proba(n["x"])[:, 1] for n in te], thr)
            r = rows[level]
            print(f"{src}->{dst} {level:17s} F1 {r['f1']:.3f} P {r['precision']:.3f} R {r['recall']:.3f} "
                  f"timing {r['median_abs_timing_error_s']:.2f}s FD/h {r['false_detections_per_hour']:.1f}")
        report["results"][f"{src}_to_{dst}"] = rows
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "report.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
