"""Offline CLAS replay with an arousal-risk veto (issue #96).

For each night: run the causal slow-oscillation trigger on C4 EEG during N2/N3,
mark clicks that land within WINDOW_S before a spontaneous arousal ("bad clicks"),
then compare vetoes that drop the same fraction of clicks:

  beats    drop clicks whose beat-level risk (last 5 RR vs prior minute, prior
           context) is highest; model trained on the *other* cohort, so no
           subject or cohort overlap with the night being replayed
  random   drop the same number of clicks at random (200 draws)

Metric: bad clicks removed per click dropped, relative to random (enrichment).
Enrichment 1.0 means the veto is no better than chance.

Usage: python scripts/run_clas_replay.py --mesa ~/data/nsrr/mesa --shhs ~/data/nsrr/shhs --out artifacts/clas_replay_dev25
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kahlus.clas_replay import clicks_before_arousal, so_clicks
from kahlus.heart_arousal import (
    COHORTS,
    EPOCH_S,
    anchor_labels,
    beat_features,
    fit_predict,
    half_night_shift,
    prior_features,
)
from kahlus.nsrr import (
    classify_arousals,
    discover_nsrr,
    load_nsrr_epoch_matrix,
    mesa_fit_allowed,
    nsrr_root,
    parse_nsrr_events,
    sleep_stage_epochs,
)
from kahlus.nsrr_cache import NightCache
from kahlus.qrs_fast import causal_beats

WINDOW_S = 3.0
VETO_FRACTIONS = (0.1, 0.2, 0.3)
DEEP = {"Stage 2 sleep", "Stage 3 sleep", "Stage 4 sleep"}
N_PRIOR = 6  # prior_features columns, then beat_features columns
# Which columns each veto model sees; "beats_shift" uses beat features rotated by half a night.
VETOES = ("prior", "beats", "prior+beats", "prior+beats_shift")


def night(rec, cohort: str) -> dict:
    labels = COHORTS[cohort]
    m = load_nsrr_epoch_matrix(rec.edf_path, channel_groups={"ecg": labels["ecg_labels"], "eeg": labels["eeg_labels"]})
    fs = m["sfreq"]
    ecg = m["epochs"]["ecg"][:, :, 0].ravel()
    eeg_uv = m["epochs"]["eeg"][:, :, 0].ravel() * 1e6
    events = parse_nsrr_events(rec.xml_path)
    n_sec = int(eeg_uv.size / fs)
    # Allowed where the last *completed* epoch was N2/N3 (what a live system knows).
    allowed = np.zeros(n_sec, dtype=bool)
    for s, e, c in sleep_stage_epochs(events):
        if c in DEEP:
            allowed[int(s + EPOCH_S) : min(int(e + EPOCH_S), n_sec)] = True
    clicks = so_clicks(eeg_uv, fs, allowed)
    onsets = np.asarray(sorted(a.onset for a in classify_arousals(events) if a.kind == "spontaneous"))
    bad = clicks_before_arousal(clicks, onsets, WINDOW_S)
    bt, rr = causal_beats(ecg, fs)
    x = np.concatenate([prior_features(events, clicks), beat_features(bt, rr, clicks)], axis=1)
    # Training rows for the risk model: the standard no-lookahead anchors on a 1 s grid.
    t, y = anchor_labels(events, total_s=n_sec, horizons=(int(WINDOW_S),), step_s=1.0)
    xt = np.concatenate([prior_features(events, t), beat_features(bt, rr, t)], axis=1)
    return {"clicks": clicks, "bad": bad, "x_click": x, "x_train": xt, "y_train": y[:, 0]}


def load(cohort: str, root: Path, cache: Path) -> list[dict]:
    recs, found = discover_nsrr(root, dataset=cohort)
    store = NightCache.bind(cache, cohort, producer="run_clas_replay", report=found, ids=[r.subject_id for r in recs])
    out = []
    for rec in recs:
        if not store.has(rec.subject_id):
            store.save(rec.subject_id, **night(rec, cohort))
        out.append(store.load(rec.subject_id))
    return out


def columns(x: np.ndarray, x_shift: np.ndarray, veto: str) -> np.ndarray:
    parts = {"prior": x[:, :N_PRIOR], "beats": x[:, N_PRIOR:], "beats_shift": x_shift[:, N_PRIOR:]}
    return np.concatenate([parts[p] for p in veto.split("+")], axis=1)


def replay(test: list[dict], train: list[dict], seed: int) -> dict:
    xtr = np.concatenate([n["x_train"] for n in train])
    xtr_s = np.concatenate([half_night_shift(n["x_train"]) for n in train])
    ytr = np.concatenate([n["y_train"] for n in train])
    keep = ytr >= 0
    xc = np.concatenate([n["x_click"] for n in test])
    xc_s = np.concatenate([half_night_shift(n["x_click"]) for n in test])
    bad = np.concatenate([n["bad"] for n in test])
    rng = np.random.default_rng(seed)
    out = {"clicks": int(bad.size), "bad_clicks": int(bad.sum()), "bad_rate": float(bad.mean())}
    for q in VETO_FRACTIONS:
        k = int(round(q * bad.size))
        rand = [int(bad[rng.choice(bad.size, k, replace=False)].sum()) for _ in range(200)]
        row = {"dropped": k, "random_mean": float(np.mean(rand)), "random_p95": float(np.percentile(rand, 95))}
        for veto in VETOES:
            risk = fit_predict(columns(xtr, xtr_s, veto)[keep], ytr[keep], columns(xc, xc_s, veto), seed)
            removed = int(bad[np.argsort(-risk)[:k]].sum())
            row[veto] = {"bad_removed": removed, "enrichment": float(removed / max(np.mean(rand), 1e-9))}
        out[str(q)] = row
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesa", type=Path, default=nsrr_root("mesa"))
    ap.add_argument("--shhs", type=Path, default=nsrr_root("shhs"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    data = {c: load(c, getattr(a, c), a.out / "cache") for c in ("mesa", "shhs")}
    report = {"window_s": WINDOW_S, "replay_on_mesa_risk_from_shhs": replay(data["mesa"], data["shhs"], 0)}
    if mesa_fit_allowed():  # final mode: held-out MESA is test-only
        report["replay_on_shhs_risk_from_mesa"] = replay(data["shhs"], data["mesa"], 0)
    for name, r in report.items():
        if isinstance(r, dict):
            print(name, f"clicks {r['clicks']} bad {r['bad_clicks']} ({r['bad_rate']:.3%})")
            for q in map(str, VETO_FRACTIONS):
                row = r[q]
                print(f"  veto {q}: random {row['random_mean']:.1f} (p95 {row['random_p95']:.0f})  " + "  ".join(
                    f"{v} {row[v]['bad_removed']} x{row[v]['enrichment']:.2f}" for v in VETOES))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "report.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
