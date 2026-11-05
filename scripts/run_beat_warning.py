"""Beat-level arousal warning at 1-5 s (issue #83, input to #96).

Same no-lookahead anchors as the main benchmark but on a 1 s grid, with
beat-resolved RR features (last 5 intervals relative to the prior minute) and a
half-night shift control. Headline: prior+beats vs prior+beats_shift.

Usage: python scripts/run_beat_warning.py --mesa ~/data/nsrr/mesa --shhs ~/data/nsrr/shhs --out artifacts/beat_warning_dev25
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_heart_arousal_benchmark import COHORTS, fit_predict  # noqa: E402

from neurotwin.adapters.nsrr import discover_nsrr_recordings, load_nsrr_epoch_matrix, parse_nsrr_events
from neurotwin.adapters.qrs_fast import causal_beats
from neurotwin.forecastability.heart_arousal import anchor_labels, beat_features, prior_features
from neurotwin.forecastability.m1 import _cluster_bootstrap_rfs, _rfs_bits

HORIZONS = (1, 2, 3, 5)
STEP = 1.0


def night(rec, cohort: str) -> dict:
    m = load_nsrr_epoch_matrix(rec.edf_path, channel_groups={"ecg": COHORTS[cohort]["ecg_labels"]})
    fs = m["sfreq"]
    ecg = m["epochs"]["ecg"][:, :, 0].ravel()
    events = parse_nsrr_events(rec.xml_path)
    t, y = anchor_labels(events, total_s=ecg.size / fs, horizons=HORIZONS, step_s=STEP)
    bt, rr = causal_beats(ecg, fs)
    beats = beat_features(bt, rr, t)
    return {"y": y, "prior": prior_features(events, t), "beats": beats, "beats_shift": np.roll(beats, beats.shape[0] // 2, axis=0)}


def load(cohort: str, root: Path, cache: Path) -> dict:
    cache.mkdir(parents=True, exist_ok=True)
    parts = []
    for i, rec in enumerate(discover_nsrr_recordings(root, dataset=cohort)):
        f = cache / f"{cohort}-{rec.subject_id}.npz"
        if not f.exists():
            np.savez(f, **night(rec, cohort))
        z = np.load(f)
        parts.append({k: z[k] for k in z.files} | {"subject": np.full(z["y"].shape[0], i)})
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def design(d: dict, model: str) -> np.ndarray:
    return np.concatenate([d[b] for b in model.split("+")], axis=1)


def run(train: dict, test: dict | None, k: int, seed: int, bootstrap: str) -> dict:
    """Subject-grouped 5-fold CV when test is None, otherwise train -> test transfer."""
    from sklearn.model_selection import GroupKFold

    models = ("prior", "prior+beats", "prior+beats_shift")
    if test is None:
        keep = train["y"][:, k] >= 0
        y, subj = train["y"][keep, k], train["subject"][keep]
        preds = {}
        for m in models:
            x, p = design(train, m)[keep], np.full(keep.sum(), np.nan)
            for tr, te in GroupKFold(5).split(x, y, subj):
                p[te] = fit_predict(x[tr], y[tr], x[te], seed)
            preds[m] = p
    else:
        a, b = train["y"][:, k] >= 0, test["y"][:, k] >= 0
        y, subj = test["y"][b, k], test["subject"][b]
        preds = {m: fit_predict(design(train, m)[a], train["y"][a, k], design(test, m)[b], seed) for m in models}
    out = {"n": int(y.size), "pos": int(y.sum())}
    for m, ref in (("prior+beats", "prior"), ("prior+beats_shift", "prior"), ("prior+beats", "prior+beats_shift")):
        r = _cluster_bootstrap_rfs(y, preds[ref], preds[m], subj, seed=seed, mode=bootstrap)
        out[f"{m} vs {ref}"] = {"bits": _rfs_bits(y, preds[ref], preds[m]), "lo": r["rfs_ci_low"], "hi": r["rfs_ci_high"]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesa", type=Path, required=True)
    ap.add_argument("--shhs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", choices=("smoke", "claim"), default="smoke", help="smoke: 300 resamples; claim: 2000")
    a = ap.parse_args()
    data = {c: load(c, getattr(a, c), a.out / "cache") for c in ("mesa", "shhs")}
    report = {}
    for k, h in enumerate(HORIZONS):
        r = report[f"h{h}"] = {
            "mesa_cv": run(data["mesa"], None, k, a.seed, a.bootstrap),
            "shhs_cv": run(data["shhs"], None, k, a.seed, a.bootstrap),
            "mesa_to_shhs": run(data["mesa"], data["shhs"], k, a.seed, a.bootstrap),
            "shhs_to_mesa": run(data["shhs"], data["mesa"], k, a.seed, a.bootstrap),
        }
        key = "prior+beats vs prior+beats_shift"
        print(f"h={h}s aligned-beat bits " + "  ".join(f"{s}: {v[key]['bits']:+.4f} [{v[key]['lo']:+.4f},{v[key]['hi']:+.4f}] pos={v['pos']}" for s, v in r.items()))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "report.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
