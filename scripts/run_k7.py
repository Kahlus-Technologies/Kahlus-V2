"""K7: do causal trigger times come before scored arousals? A replay measurement, not a causal result about clicks.

For every would-be click of the causal slow-oscillation trigger (kahlus.clas_replay), count whether a spontaneous
arousal starts in the 3 s after it, and compare with 5 matched control moments from the same night (same stage,
same time-since-arousal bin, within 600 s, no click within 3 s). Relative risk and a heart-rate outcome that needs
no scorer, with 95% intervals that resample whole subjects. Rules, primary cell and gate are frozen in
docs/results/k7_k1_s1_preregistration.md (section 1); the other settings and sensitivity runs are secondary.

Usage (dev slice, pilot only):
  python scripts/run_k7.py --mesa ~/data/nsrr/mesa --shhs ~/data/nsrr/shhs --out artifacts/k7_dev
Confirmatory SHHS-1 run (no MESA, dev nights 200001-200025 left out):
  python scripts/run_k7.py --shhs $SHHS --exclude-dev-shhs --workers 64 --out artifacts/k7_shhs1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from kahlus.clas_replay import (
    MEAN_DIFFERENCE,
    RELATIVE_RISK,
    clicks_before_arousal,
    first_of_runs,
    hr_change,
    matched_controls,
    rotate_clicks,
    running_level,
    since_arousal_bin,
    so_clicks,
    so_filter,
    subject_bootstrap,
)
from kahlus.forecast_bench import is_tuning
from kahlus.heart_arousal import COHORTS, EPOCH_S
from kahlus.nsrr import (
    classify_arousals,
    discover_nsrr,
    load_nsrr_epoch_matrix,
    nsrr_root,
    parse_nsrr_events,
    sleep_stage_epochs,
)
from kahlus.nsrr_cache import NightCache, git_state
from kahlus.qrs_fast import causal_beats

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "docs/results/k7_k1_s1_preregistration.md"
STAGE_CODE = {"Stage 2 sleep": 2, "Stage 3 sleep": 3, "Stage 4 sleep": 3}  # N3 includes stages 3 and 4
# Trigger settings: level in microvolts and the stage codes where the trigger may fire. "primary" is the repo default.
# Not run: the "second crossing within 2.5 s" variant of the preregistered grid (ceiling of this pilot).
SETTINGS = {
    "primary": (-40.0, (2, 3)), "t40_n3": (-40.0, (3,)), "t60_n23": (-60.0, (2, 3)),
    "t60_n3": (-60.0, (3,)), "t80_n23": (-80.0, (2, 3)), "t80_n3": (-80.0, (3,)),
}
PCT = 90.0  # running percentile of |filtered EEG| so far, for the amplitude-following threshold
# Sensitivity variants of the primary setting (all secondary): pct = percentile threshold, first = first click of
# each run, anyctrl = controls without the "no click within 3 s" rule, rot = clicks rotated by half the night
# (the validity gate), seedN = other control draws.
SENSITIVITY = ("pct", "first", "anyctrl", "rot", "seed1", "seed2", "seed3", "seed4")
VARIANTS = (*SETTINGS, *SENSITIVITY)
OUTCOMES = ("sp3", "hr", "sp1", "sp5", "sp10", "any3")  # sp = spontaneous arousal onset within w s after; any = any arousal
SETTING_NAMES = tuple(v for v in SETTINGS if v != "primary")
MAX_DROPPED = 0.05


def outcomes(t: np.ndarray, sp: np.ndarray, any_on: np.ndarray, bt: np.ndarray, rr: np.ndarray) -> dict[str, np.ndarray]:
    ok = np.isfinite(t)
    flat = np.where(ok, t, 0.0).ravel()
    res = {f"sp{w}": clicks_before_arousal(flat, sp, w).astype(float) for w in (1, 3, 5, 10)}
    res["any3"] = clicks_before_arousal(flat, any_on, 3.0).astype(float)
    res["hr"] = hr_change(bt, rr, flat)
    return {k: np.where(ok.ravel(), v, np.nan).reshape(t.shape) for k, v in res.items()}


def counts(clicks: np.ndarray, ctrl: np.ndarray, **kw) -> np.ndarray:
    """(outcome, [events at clicks, events at controls, matched clicks]) summed over one night. Each matched
    click contributes its controls' mean, so the controls of a click weigh 1 in total."""
    c, k = outcomes(clicks, **kw), outcomes(ctrl, **kw)
    out = np.zeros((len(OUTCOMES), 3))
    has = np.isfinite(ctrl).any(axis=1)
    for j, o in enumerate(OUTCOMES):
        fin = np.isfinite(k[o])
        kmean = np.where(fin.any(axis=1), np.where(fin, k[o], 0.0).sum(axis=1) / np.maximum(fin.sum(axis=1), 1), np.nan)
        m = has & np.isfinite(c[o]) & np.isfinite(kmean)
        out[j] = c[o][m].sum(), kmean[m].sum(), m.sum()
    return out


def night(rec, cohort: str, store: NightCache, control_seed: int = 0) -> None:
    if store.has(rec.subject_id):
        return
    labels = COHORTS[cohort]
    m = load_nsrr_epoch_matrix(rec.edf_path, channel_groups={"ecg": labels["ecg_labels"], "eeg": labels["eeg_labels"]})
    fs = m["sfreq"]
    ecg = m["epochs"]["ecg"][:, :, 0].ravel()
    eeg_uv = m["epochs"]["eeg"][:, :, 0].ravel() * 1e6
    events = parse_nsrr_events(rec.xml_path)
    n_sec = int(eeg_uv.size / fs)
    # Stage code where the last *completed* epoch was N2/N3 (what a live system knows); 0 = trigger off.
    stage_sec = np.zeros(n_sec, dtype=np.int8)
    for s, e, c in sleep_stage_epochs(events):
        if c in STAGE_CODE:
            stage_sec[int(s + EPOCH_S) : min(int(e + EPOCH_S), n_sec)] = STAGE_CODE[c]
    arousals = classify_arousals(events)
    sp = np.sort([a.onset for a in arousals if a.kind == "spontaneous"])
    any_on = np.sort([a.onset for a in arousals])
    bin_sec = since_arousal_bin(any_on, n_sec)
    bt, rr = causal_beats(ecg, fs)
    kw = {"sp": sp, "any_on": any_on, "bt": bt, "rr": rr}
    base = {name: so_clicks(eeg_uv, fs, np.isin(stage_sec, stages), neg) for name, (neg, stages) in SETTINGS.items()}
    primary = base["primary"]
    allowed = stage_sec > 0
    level = running_level(so_filter(eeg_uv, fs), fs, PCT)
    rotated = rotate_clicks(primary, stage_sec)
    spec = {  # name -> (clicks, avoid, match_bin, control seed)
        **{v: (c, None, True, control_seed) for v, c in base.items()},
        "pct": (so_clicks(eeg_uv, fs, allowed, level), None, True, control_seed),
        "first": (first_of_runs(primary), primary, True, control_seed),
        "anyctrl": (primary, np.zeros(0), True, control_seed),
        "rot": (rotated, np.sort(np.concatenate([primary, rotated])), True, control_seed),
        **{f"seed{i}": (primary, None, True, i) for i in (1, 2, 3, 4)},
    }
    out = np.zeros((len(VARIANTS), len(OUTCOMES), 3))
    n_clicks, dropped = np.zeros(len(VARIANTS), dtype=int), np.zeros(len(VARIANTS), dtype=int)
    for i, v in enumerate(VARIANTS):
        clicks, avoid, match_bin, seed = spec[v]
        rng = np.random.default_rng([seed, zlib.crc32(rec.subject_id.encode())])
        ctrl, dropped[i] = matched_controls(clicks, stage_sec, bin_sec, rng=rng, avoid=avoid, match_bin=match_bin)
        n_clicks[i] = clicks.size
        out[i] = counts(clicks, ctrl, **kw)
    store.save(rec.subject_id, counts=out, n_clicks=n_clicks, dropped=dropped, n_sec=np.array(n_sec))


def discover(cohort: str, root: Path, cache: Path, exclude_dev: bool) -> tuple[list, NightCache]:
    recs, found = discover_nsrr(root, dataset=cohort)
    if exclude_dev and cohort == "shhs":
        dev = {Path(p).stem.split("-")[-1] for p in (REPO / "docs/results/provenance/shhs_dev_slice_edfs.txt").read_text().split()}
        recs = [r for r in recs if r.subject_id not in dev]
    store = NightCache.bind(cache, cohort, producer="run_k7", report=found, ids=[r.subject_id for r in recs])
    return recs, store


def parity(subject_id: str) -> int:
    return int(re.findall(r"\d+", subject_id)[-1]) % 2


def table(nights: list[dict], cohort: str, n_boot: int) -> list[dict]:
    """Estimates per variant, outcome and subset of subjects; the interval resamples whole subjects."""
    subsets = {
        "all": lambda n: True,
        "tuning": lambda n: is_tuning({"dataset": cohort, "subject": n["subject"]}),
        "other": lambda n: not is_tuning({"dataset": cohort, "subject": n["subject"]}),
        "odd": lambda n: parity(n["subject"]) == 1,
        "even": lambda n: parity(n["subject"]) == 0,
    }
    rows = []
    for sname, pick in subsets.items():
        sel = [n for n in nights if pick(n)]
        if not sel:
            continue
        for vi, v in enumerate(VARIANTS):
            for oi, o in enumerate(OUTCOMES):
                if (sname == "all" and (o in ("sp3", "hr") or v == "primary")) or (sname != "all" and (o == "sp3" or (o == "hr" and v == "primary"))):
                    c = np.stack([n["counts"][vi, oi] for n in sel])
                    r = subject_bootstrap(c, MEAN_DIFFERENCE if o == "hr" else RELATIVE_RISK, n_boot=n_boot, seed=0)
                    rows.append({"cohort": cohort, "subset": sname, "variant": v, "outcome": o, "n_subjects": len(sel),
                                 "matched_clicks": int(c[:, 2].sum()), **r,
                                 "rate_click": float(c[:, 0].sum() / c[:, 2].sum()), "rate_control": float(c[:, 1].sum() / c[:, 2].sum())})
    # Holm over the other trigger settings, on all subjects, primary outcome.
    sec = sorted((r for r in rows if r["subset"] == "all" and r["outcome"] == "sp3" and r["variant"] in SETTING_NAMES), key=lambda r: r["p"])
    run = 0.0
    for k, r in enumerate(sec):
        run = max(run, min(1.0, (len(sec) - k) * r["p"]))
        r["p_holm"] = run
    return rows


def find(rows, **kw):
    return next(r for r in rows if all(r[k] == v for k, v in kw.items()))


def verdicts(rows: list[dict], clicks: dict) -> dict:
    """Gate, matching check and the four-part finding rule of the preregistration, for one cohort."""
    rot = find(rows, subset="all", variant="rot", outcome="sp3")
    prim = find(rows, subset="all", variant="primary", outcome="sp3")
    dropped = clicks["primary"]["dropped_fraction"]
    gate = bool(rot["ci_low"] <= 1.0 <= rot["ci_high"])
    halves = [find(rows, subset=h, variant="primary", outcome="sp3") for h in ("odd", "even")]
    direction = np.sign(np.log(prim["estimate"]))
    tuning = [r for r in rows if r["subset"] == "tuning" and r["variant"] in SETTINGS and r["outcome"] == "sp3"]
    return {
        "rotated_gate_passes": gate,
        "matching_ok_dropped_fraction": dropped, "matching_ok": bool(dropped <= MAX_DROPPED),
        "reportable_as_primary": bool(gate and dropped <= MAX_DROPPED),
        "finding": {
            "interval_excludes_1": bool(prim["ci_low"] > 1 or prim["ci_high"] < 1),
            "effect_at_least_10_percent": bool(prim["estimate"] >= 1.10 or prim["estimate"] <= 0.91),
            "rotated_gate": gate,
            "halves_same_direction_excluding_1": bool(all(np.sign(np.log(h["estimate"])) == direction and (h["ci_low"] > 1 or h["ci_high"] < 1) for h in halves)),
        },
        "kill_rule_all_settings_within_0.9_1.1_on_tuning": bool(tuning and all(0.9 <= r["estimate"] <= 1.1 for r in tuning)),
    }


def clicks_summary(nights: list[dict]) -> dict:
    out = {}
    for vi, v in enumerate(VARIANTS):
        n = np.array([x["n_clicks"][vi] for x in nights])
        d = np.array([x["dropped"][vi] for x in nights])
        out[v] = {"clicks_per_night_mean": float(n.mean()), "clicks_per_night_median": float(np.median(n)), "clicks_per_night_min": int(n.min()),
                  "clicks_per_night_max": int(n.max()), "clicks": int(n.sum()), "dropped": int(d.sum()), "dropped_fraction": float(d.sum() / max(n.sum(), 1))}
    return out


def markdown(report: dict) -> str:
    L = []
    for cohort, rows in report["tables"].items():
        v = report["verdicts"][cohort]
        L += [f"### {cohort.upper()}: {report['subjects'][cohort]} nights", "",
              f"Rotated-trigger gate: {'PASS' if v['rotated_gate_passes'] else 'FAIL'}. Clicks dropped by matching: {v['matching_ok_dropped_fraction']:.2%}"
              f" ({'ok' if v['matching_ok'] else 'matching failed'}).", "",
              "| setting | clicks/night (median) | click rate | control rate | relative risk [95% CI] | Holm p |", "|---|---|---|---|---|---|"]
        for name in VARIANTS:
            r = find(rows, subset="all", variant=name, outcome="sp3")
            cs = report["clicks"][cohort][name]
            holm = f"{r['p_holm']:.3f}" if "p_holm" in r else ""
            L.append(f"| {name}{' (primary)' if name == 'primary' else ''} | {cs['clicks_per_night_median']:.0f} | {r['rate_click']:.4f} | {r['rate_control']:.4f} "
                     f"| {r['estimate']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}] | {holm} |")
        L += ["", "Primary setting, other windows and the heart-rate outcome (secondary):", "", "| outcome | click | control | estimate [95% CI] |", "|---|---|---|---|"]
        for o in OUTCOMES:
            r = find(rows, subset="all", variant="primary", outcome=o)
            kind = "mean bpm difference" if o == "hr" else "relative risk"
            L.append(f"| {o} ({kind}) | {r['rate_click']:.4f} | {r['rate_control']:.4f} | {r['estimate']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}] |")
        L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mesa", type=Path, default=None)
    ap.add_argument("--shhs", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--exclude-dev-shhs", action="store_true", help="leave SHHS-1 nights 200001-200025 out (confirmatory runs)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--bootstrap", type=int, default=2000)
    a = ap.parse_args()
    roots = {c: p for c, p in (("mesa", a.mesa), ("shhs", a.shhs)) if p is not None} or {"mesa": nsrr_root("mesa"), "shhs": nsrr_root("shhs")}
    a.out.mkdir(parents=True, exist_ok=True)
    starts = a.out / "starts.txt"
    n_starts = len(starts.read_text().split()) + 1 if starts.exists() else 1
    starts.write_text(starts.read_text() + "x " if starts.exists() else "x ")
    report: dict = {"tables": {}, "clicks": {}, "verdicts": {}, "subjects": {}}
    manifests = {}
    for cohort, root in roots.items():
        recs, store = discover(cohort, root, a.out / "cache", a.exclude_dev_shhs)
        done = partial(night, cohort=cohort, store=store)
        if a.workers > 1:
            with ProcessPoolExecutor(a.workers) as ex:
                list(ex.map(done, recs, chunksize=4))
        else:
            for r in recs:
                done(r)
        nights = [{"subject": r.subject_id, **store.load(r.subject_id)} for r in recs]
        manifests[cohort] = store.identity["ids_sha256"]
        report["subjects"][cohort] = len(nights)
        report["clicks"][cohort] = clicks_summary(nights)
        report["tables"][cohort] = table(nights, cohort, a.bootstrap)
        report["verdicts"][cohort] = verdicts(report["tables"][cohort], report["clicks"][cohort])
    config = {"settings": SETTINGS, "variants": VARIANTS, "outcomes": OUTCOMES, "pct": PCT, "bootstrap": a.bootstrap, "control_seed": 0, "bootstrap_seed": 0}
    import scipy
    report["provenance"] = {
        **{k: v for k, v in git_state().items()}, "commit_env": os.environ.get("K7_COMMIT"), "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(), "config": config,
        "data_manifest_sha256": manifests, "numpy": np.__version__, "scipy": scipy.__version__, "n_starts": n_starts,
        "preregistration_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(), "excluded_dev_shhs": a.exclude_dev_shhs,
        "note": "replay measurement, not a causal result about clicks; SHHS 'spontaneous' means not respiratory (no leg channel)",
    }
    (a.out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    (a.out / "report.md").write_text(markdown(report))
    print(markdown(report))
    print(json.dumps(report["verdicts"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
