"""Heart-only arousal forecasting benchmark (issue #83/#84).

Builds per-night features (cached as .npz), then for each horizon compares nested
models with subject-held-out CV inside each cohort and dataset-held-out transfer
(train one cohort, test the other):

  prior          time since last arousal, time since sleep onset, last stage
  prior+cardiac  + ECG RR features          <- the claim: does the heart add anything?
  prior+eeg      + EEG band power           (reference)
  prior+cardiac+eeg                         (upper bound)
  prior+ppg      MESA only (SHHS has no PPG)

Primary metric: residual forecastability in bits of prior+cardiac over prior,
with a subject-clustered bootstrap CI. AUROC and AUPRC are reported alongside.

Usage:
  python scripts/run_heart_arousal_benchmark.py --mesa ~/data/nsrr/mesa --shhs ~/data/nsrr/shhs \
      --out artifacts/heart_arousal_dev25 [--max-nights 25] [--bootstrap claim]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from neurotwin.adapters.nsrr import discover_nsrr_recordings
from neurotwin.forecastability.heart_arousal import HORIZONS_S, build_recording
from neurotwin.forecastability.m1 import _cluster_bootstrap_rfs, _rfs_bits

COHORTS = {
    "mesa": {"ecg_labels": ("EKG",), "eeg_labels": ("EEG3",), "ppg_labels": ("Pleth",)},
    "shhs": {"ecg_labels": ("ECG",), "eeg_labels": ("EEG",), "ppg_labels": ()},
}
MODELS = {
    "prior": ("prior",),
    "prior+cardiac": ("prior", "cardiac"),
    "prior+eeg": ("prior", "eeg"),
    "prior+cardiac+eeg": ("prior", "cardiac", "eeg"),
    "prior+ppg": ("prior", "ppg"),
}


SIGNAL_BLOCKS = ("cardiac", "eeg", "ppg")  # per-night z-scored; "prior" is already in shared units


def per_night_z(x: np.ndarray) -> np.ndarray:
    """Z-score each column within one night so amplifier gain and resting HR stop
    acting as cohort fingerprints. Uses the whole night, i.e. a small amount of
    within-night look-ahead in the normalizer; that is the same for every model
    being compared, so it cannot favor one block over another."""
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    return (x - mu) / np.where(sd > 0, sd, 1.0)


CAUSAL_WARMUP = 60  # anchors (5 min at 5 s spacing) before running stats are trusted


def causal_z(x: np.ndarray) -> np.ndarray:
    """Z-score each row against the mean and SD of earlier rows in the same night
    only (expanding window, NaN-aware). Rows before CAUSAL_WARMUP become NaN.
    Anchors are time-ordered, so row i never sees row i or later."""
    ok = ~np.isnan(x)
    v = np.where(ok, x, 0.0).astype(np.float64)
    n = np.cumsum(ok, axis=0) - ok  # count of earlier non-NaN rows
    s1 = np.cumsum(v, axis=0) - v
    s2 = np.cumsum(v * v, axis=0) - v * v
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = s1 / n
        sd = np.sqrt(np.maximum(s2 / n - mu * mu, 0.0))
        z = (x - mu) / np.where(sd > 0, sd, np.nan)
    z[n < CAUSAL_WARMUP] = np.nan
    return z.astype(np.float32)


NORMALIZERS = {"none": lambda x: x, "night": per_night_z, "causal": causal_z}


def load_cohort(name: str, root: Path, cache: Path, max_nights: int | None, normalize: str) -> dict:
    recs = discover_nsrr_recordings(root, dataset=name)[:max_nights]
    cache.mkdir(parents=True, exist_ok=True)
    nights = []
    for rec in recs:
        f = cache / f"{name}-{rec.subject_id}.npz"
        if not f.exists():
            try:
                r = build_recording(rec.edf_path, rec.xml_path, **COHORTS[name])
            except ValueError as exc:
                print(f"skip {name} {rec.subject_id}: {exc}", file=sys.stderr)
                continue
            np.savez(f, anchors=r["anchors"], labels=r["labels"], **{f"x_{k}": v for k, v in r["blocks"].items()})
        nights.append((rec.subject_id, np.load(f)))
    blocks = {k[2:] for _, z in nights for k in z.files if k.startswith("x_")}
    norm = NORMALIZERS[normalize]
    prep = lambda b, v: norm(v) if b in SIGNAL_BLOCKS else v  # noqa: E731
    return {
        "subject": np.concatenate([np.full(z["anchors"].size, i) for i, (_, z) in enumerate(nights)]),
        "labels": np.concatenate([z["labels"] for _, z in nights]),
        "x": {b: np.concatenate([prep(b, z[f"x_{b}"]) for _, z in nights]) for b in blocks},
        "n_nights": len(nights),
    }


def fit_predict(x_tr, y_tr, x_te, seed: int) -> np.ndarray:
    # ponytail: one fixed gradient-boosting config; a tuned or sequence model is the upgrade path.
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=seed)
    clf.fit(x_tr, y_tr)
    return clf.predict_proba(x_te)[:, 1]


def design(data: dict, blocks: tuple[str, ...]) -> np.ndarray | None:
    if any(b not in data["x"] for b in blocks):
        return None
    return np.concatenate([data["x"][b] for b in blocks], axis=1)


def within_cohort(data: dict, k: int, seed: int) -> dict[str, np.ndarray]:
    """Out-of-fold predictions from 5-fold subject-grouped CV."""
    y = data["labels"][:, k]
    keep = y >= 0
    preds = {}
    for name, blocks in MODELS.items():
        x = design(data, blocks)
        if x is None:
            continue
        p = np.full(keep.sum(), np.nan)
        for tr, te in GroupKFold(n_splits=5).split(x[keep], y[keep], data["subject"][keep]):
            p[te] = fit_predict(x[keep][tr], y[keep][tr], x[keep][te], seed)
        preds[name] = p
    return {"y": y[keep], "subject": data["subject"][keep], "preds": preds}


def transfer(train: dict, test: dict, k: int, seed: int) -> dict[str, np.ndarray]:
    ytr, yte = train["labels"][:, k], test["labels"][:, k]
    a, b = ytr >= 0, yte >= 0
    preds = {}
    for name, blocks in MODELS.items():
        xtr, xte = design(train, blocks), design(test, blocks)
        if xtr is None or xte is None:
            continue
        preds[name] = fit_predict(xtr[a], ytr[a], xte[b], seed)
    return {"y": yte[b], "subject": test["subject"][b], "preds": preds}


def score(run: dict, seed: int, bootstrap: str) -> dict:
    y, subj, preds = run["y"], run["subject"], run["preds"]
    out = {"n_windows": int(y.size), "n_positive": int(y.sum()), "base_rate": float(y.mean())}
    for name, p in preds.items():
        out[name] = {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p))}
    for name in ("prior+cardiac", "prior+eeg", "prior+cardiac+eeg", "prior+ppg"):
        if name in preds:
            rfs = _cluster_bootstrap_rfs(y, preds["prior"], preds[name], subj, seed=seed, mode=bootstrap)
            rfs["rfs_bits"] = _rfs_bits(y, preds["prior"], preds[name])
            out[name]["vs_prior"] = rfs
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesa", type=Path, required=True)
    ap.add_argument("--shhs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-nights", type=int, default=None)
    ap.add_argument("--bootstrap", choices=("smoke", "claim"), default="smoke")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--normalize", choices=tuple(NORMALIZERS), default="none",
                    help="cardiac/EEG/PPG features: raw, z-scored over the whole night, or causal expanding z-score")
    ap.add_argument("--cache", type=Path, default=None, help="feature cache dir (default: OUT/cache); raw features, safe to share across runs")
    a = ap.parse_args()
    cache = a.cache or a.out / "cache"
    data = {
        "mesa": load_cohort("mesa", a.mesa, cache, a.max_nights, a.normalize),
        "shhs": load_cohort("shhs", a.shhs, cache, a.max_nights, a.normalize),
    }
    report = {"horizons_s": list(HORIZONS_S), "normalize": a.normalize, "nights": {k: v["n_nights"] for k, v in data.items()}, "results": {}}
    for k, h in enumerate(HORIZONS_S):
        r = report["results"][f"h{h}"] = {}
        for c in ("mesa", "shhs"):
            r[f"{c}_cv"] = score(within_cohort(data[c], k, a.seed), a.seed, a.bootstrap)
        r["mesa_to_shhs"] = score(transfer(data["mesa"], data["shhs"], k, a.seed), a.seed, a.bootstrap)
        r["shhs_to_mesa"] = score(transfer(data["shhs"], data["mesa"], k, a.seed), a.seed, a.bootstrap)
        print(f"h={h:>3}s " + "  ".join(
            f"{split}: card {v['prior+cardiac']['vs_prior']['rfs_bits']:+.4f} [{v['prior+cardiac']['vs_prior']['rfs_ci_low']:+.4f},{v['prior+cardiac']['vs_prior']['rfs_ci_high']:+.4f}]"
            for split, v in r.items()
        ))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "report.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
