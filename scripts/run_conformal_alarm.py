"""Conformal alarm budget on the arousal forecaster (issue #89).

Within a cohort: subjects split 50/25/25 into train / calibrate / test (5 seeds).
Across cohorts: train + calibrate on one cohort, test on the other, which is
where exchangeability should break. For each alpha we report the realized miss
rate overall and per test subject, and the false alarms per hour it costs.

Usage: python scripts/run_conformal_alarm.py --cache artifacts/heart_arousal_dev25/cache \
    --out artifacts/conformal_alarm_dev25 [--horizon 30] [--model prior+cardiac]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_heart_arousal_benchmark import MODELS, design  # noqa: E402

from neurotwin.adapters.nsrr import gate_mesa_ids, mesa_fit_allowed
from neurotwin.forecastability.conformal import alarm_metrics, miss_rate_threshold
from neurotwin.forecastability.heart_arousal import HORIZONS_S, STEP_S, causal_z, fit_predict

ALPHAS = (0.1, 0.2, 0.3, 0.5)


def load(cache: Path, cohort: str, k: int) -> dict:
    """Per-subject arrays from the benchmark cache, causal-normalized, label column k."""
    subs = []
    files = sorted(cache.glob(f"{cohort}-*.npz"))
    if cohort == "mesa":  # caches go through the same MESA seal as raw files
        keep = set(gate_mesa_ids([f.stem.split("-", 1)[1] for f in files]))
        files = [f for f in files if f.stem.split("-", 1)[1] in keep]
    for f in files:
        z = np.load(f)
        y = z["labels"][:, k]
        keep = y >= 0
        x = {b[2:]: (causal_z(z[b]) if b[2:] != "prior" else z[b])[keep] for b in z.files if b.startswith("x_")}
        subs.append({"y": y[keep], "x": x})
    return subs


def stack(subs: list[dict], idx) -> dict:
    return {
        "y": np.concatenate([subs[i]["y"] for i in idx]),
        "subject": np.concatenate([np.full(subs[i]["y"].size, i) for i in idx]),
        "x": {b: np.concatenate([subs[i]["x"][b] for i in idx]) for b in subs[idx[0]]["x"]},
    }


def evaluate(train: dict, cal: dict, test: dict, blocks: tuple[str, ...], seed: int) -> dict:
    model_in = lambda d: design({"x": d["x"]}, blocks)  # noqa: E731
    xtr, xc, xt = model_in(train), model_in(cal), model_in(test)
    clf_scores_cal = fit_predict(xtr, train["y"], np.concatenate([xc, xt]), seed)
    s_cal, s_test = clf_scores_cal[: xc.shape[0]], clf_scores_cal[xc.shape[0] :]
    out = {}
    for alpha in ALPHAS:
        thr = miss_rate_threshold(s_cal[cal["y"] == 1], alpha)
        m = alarm_metrics(s_test, test["y"], thr, step_s=STEP_S)
        per = [
            1.0 - np.mean(s_test[(test["subject"] == s) & (test["y"] == 1)] >= thr)
            for s in np.unique(test["subject"])
            if ((test["subject"] == s) & (test["y"] == 1)).sum() >= 5
        ]
        m["per_subject_miss"] = per
        m["per_subject_miss_p10_p90"] = [float(np.percentile(per, 10)), float(np.percentile(per, 90))] if per else None
        out[str(alpha)] = m
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--model", default="prior+cardiac", choices=tuple(MODELS))
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    k = HORIZONS_S.index(a.horizon)
    blocks = MODELS[a.model]
    data = {c: load(a.cache, c, k) for c in ("mesa", "shhs")}
    report = {"horizon_s": a.horizon, "model": a.model, "alphas": list(ALPHAS), "within": {}, "transfer": {}}
    for c, subs in data.items():
        if c == "mesa" and not mesa_fit_allowed():
            continue  # final mode: held-out MESA is test-only
        runs = []
        for seed in range(a.seeds):
            perm = np.random.default_rng(seed).permutation(len(subs))
            n = len(subs)
            tr, ca, te = perm[: n // 2], perm[n // 2 : 3 * n // 4], perm[3 * n // 4 :]
            runs.append(evaluate(stack(subs, tr), stack(subs, ca), stack(subs, te), blocks, seed))
        def pooled_spread(al: str) -> list[float] | None:
            # Per-subject miss rates from every seed's test split, pooled.
            per = [v for r in runs for v in r[al]["per_subject_miss"]]
            return [float(np.percentile(per, 10)), float(np.percentile(per, 90))] if per else None

        report["within"][c] = {
            al: {m: float(np.mean([r[al][m] for r in runs])) for m in ("miss_rate", "false_alarms_per_hour")}
            | {"per_subject_miss_p10_p90": pooled_spread(al)}
            for al in runs[0]
        }
    for src, dst in (("mesa", "shhs"), ("shhs", "mesa")):
        if src == "mesa" and not mesa_fit_allowed():
            continue
        subs = data[src]
        perm = np.random.default_rng(0).permutation(len(subs))
        tr, ca = perm[: 2 * len(subs) // 3], perm[2 * len(subs) // 3 :]
        report["transfer"][f"{src}_to_{dst}"] = evaluate(stack(subs, tr), stack(subs, ca), stack(data[dst], list(range(len(data[dst])))), blocks, 0)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "report.json").write_text(json.dumps(report, indent=2))
    for name, block in [*(("within " + c, v) for c, v in report["within"].items()), *(("transfer " + c, v) for c, v in report["transfer"].items())]:
        print(name.ljust(22), "  ".join(f"a={al}: miss {v['miss_rate']:.3f} FA/h {v['false_alarms_per_hour']:.0f}" for al, v in block.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
