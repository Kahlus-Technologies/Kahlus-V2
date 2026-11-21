"""E1 readouts (docs/prereg/e1.md): anchors, frozen-probe embeddings, probes, gate G0 and the report. Reuses scripts/pilot_ab.py.

  anchors  every 30 s epoch boundary: a 64 s window ending at least 2 s before it, the detect label, the stage label, the split
  embed    last-second embeddings of a checkpoint's student, of random weights (--random SEED --like CKPT), or band power (--bandpower)
  probe    logistic regression on an embedding file: per-subject test log loss for detect and stage -> probe-<name>.npz
  g0       gate G0 on one checkpoint: embed + probe it, random weights seeds 0-2 and band power, then the verdict
  report   per-run probes of the grid -> results table, primary D with its interval, secondary cells (refuses if a planned run is missing)

detect (primary): 1 if a scored arousal overlaps the last 10 s of the window, 0 if no arousal lies within 30 s of the window's end, else
left out; sleep epochs only. stage: 5 classes as in pilot B. Test = is_heldout subjects; tune = the 100 is_tuning subjects with the smallest
u; fit = the 300 other non-held-out subjects with the smallest u; dev IDs 200001-200025 never.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pilot_ab", HERE / "pilot_ab.py")
pa = importlib.util.module_from_spec(spec)
sys.modules["pilot_ab"] = pa
spec.loader.exec_module(pa)
spec = importlib.util.spec_from_file_location("e1_data", HERE / "e1_data.py")
e1d = importlib.util.module_from_spec(spec)
sys.modules["e1_data"] = e1d
spec.loader.exec_module(e1d)

from kahlus.forecast_bench import is_tuning  # noqa: E402
from kahlus.nsrr import SLEEP_STAGES, event_kind, parse_nsrr_events, sleep_stage_epochs, stage_at  # noqa: E402
from kahlus.shards import WARMUP_PATCHES, ShardSet, is_heldout  # noqa: E402

LN2 = math.log(2)
N_FIT, N_TUNE = 300, 100


def one_night(job):
    night, xml = job
    n = night["n"]
    events = parse_nsrr_events(xml)
    stages = sleep_stage_epochs(events)
    arousals = [(s, s + d) for c, s, d in events if event_kind(c) == "arousal"]
    t = np.arange(90.0, n + WARMUP_PATCHES, 30.0)
    j = np.asarray([pa.window_end(x) for x in t], dtype=np.int64)
    rows = []
    for ti, ji in zip(t, j):
        if ji < pa.CTX - 1 or ji >= n:
            continue
        st = stage_at(stages, ti - 15.0)
        st = "Stage 3 sleep" if st == "Stage 4 sleep" else st
        if st not in pa.STAGES:
            continue
        end = ji + WARMUP_PATCHES + 1  # the window covers seconds [end - 64, end)
        if st in SLEEP_STAGES:
            hit = any(s < end and e > end - 10 for s, e in arousals)
            near = any(s < end + 30 and e > end - 30 for s, e in arousals)
            detect = 1 if hit else (-1 if near else 0)
        else:
            detect = -1
        rows.append((ti, ji, detect, pa.STAGES.index(st)))
    return rows


def cmd_anchors(a):
    shards = ShardSet(a.shards)
    pool = [n for n in shards.nights if n["dataset"] == "shhs" and int("".join(c for c in n["subject"] if c.isdigit())) not in pa.DEV_IDS]
    held = [n for n in pool if is_heldout("shhs", n["subject"])]
    rest = sorted((n for n in pool if not is_heldout("shhs", n["subject"])), key=lambda n: e1d.u_of(n["subject"]))
    tune = [n for n in rest if is_tuning(n)][:N_TUNE]
    fit = [n for n in rest if not is_tuning(n)][:N_FIT]
    chosen = [(n, 2) for n in held] + [(n, 1) for n in tune] + [(n, 0) for n in fit]
    xml_dir = Path(a.nsrr) / "annotations-events-nsrr/shhs1"
    with Pool(max(1, a.workers)) as p:
        res = p.map(one_night, [(n, xml_dir / f"shhs1-{n['subject']}-nsrr.xml") for n, _ in chosen], chunksize=4)
    cols = {k: [] for k in ("subj", "t", "j", "detect", "stage", "split")}
    for i, ((n, sp), rows) in enumerate(zip(chosen, res)):
        for t, j, d, s in rows:
            for k, v in zip(cols, (i, t, j, d, s, sp)):
                cols[k].append(v)
    dt = dict(subj=np.int32, t=np.float32, j=np.int32, detect=np.int8, stage=np.int8, split=np.int8)
    out = Path(a.work)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "anchors.npz", subjects=np.asarray([n["subject"] for n, _ in chosen]), **{k: np.asarray(v, dtype=dt[k]) for k, v in cols.items()})
    z = {k: np.asarray(v) for k, v in cols.items()}
    test = z["split"] == 2
    ts = len(set(z["subj"][test]))
    pos = int(((z["detect"] == 1) & test).sum())
    print(f"subjects fit {len(fit)}, tune {len(tune)}, test {len(held)}; anchors {len(z['t'])}; test subjects with anchors {ts}; "
          f"test detect positives {pos}; stage counts {np.bincount(z['stage'][test], minlength=5).tolist()}", flush=True)
    print(f"ANCHORS_GATE {'PASS' if ts >= 400 and pos >= 2000 else 'FAIL'} (needs 400 test subjects and 2,000 positives)", flush=True)


class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def cmd_embed(a):
    import torch

    shards = ShardSet(a.shards)
    an = np.load(Path(a.work) / "anchors.npz")
    by_subject = {n["subject"]: n for n in shards.nights}
    nights = [by_subject[s] for s in an["subjects"]]
    ds = pa.Windows(shards, nights, an["subj"], an["j"])
    dl = torch.utils.data.DataLoader(ds, batch_size=1024, num_workers=a.workers)
    if a.bandpower:
        feats = [pa.band_features(x.numpy()) for x, _ in dl]
    else:
        if a.random is not None:
            model = pa.make_model("random", a.random, a.like)
        else:
            model = pa.make_model("pretrained", 0, a.ckpt)
        model = model.to(pa.device()).eval()
        with torch.no_grad(), torch.autocast(pa.device(), dtype=torch.bfloat16):
            feats = [model.encode(x.to(pa.device())).float().flatten(1).cpu().numpy().astype(np.float16) for x, _ in dl]
    np.save(a.out, np.concatenate(feats))
    print("embedded", a.out, flush=True)


def stage_losses(F, y, split, subj, n_subj):
    """Per-subject test log loss (nats) of a multinomial logistic regression, C from {0.01, 0.1, 1} by tune log loss."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    pick = lambda sp, cap: rng.permutation(np.flatnonzero(split == sp))[:cap]  # noqa: E731
    fit, tune, test = pick(0, 100_000), pick(1, 100_000), np.flatnonzero(split == 2)
    mu, sd = F[fit].mean(0), F[fit].std(0) + 1e-6
    Z = lambda r: ((F[r] - mu) / sd).astype(np.float32)  # noqa: E731

    def nll(m, rows):
        p = m.predict_proba(Z(rows))[np.arange(len(rows)), np.searchsorted(m.classes_, y[rows])]
        return -np.log(np.clip(p, 1e-6, 1))

    best = min((float(nll(m, tune).mean()), c) for c in (0.01, 0.1, 1.0) for m in [LogisticRegression(C=c, max_iter=300).fit(Z(fit), y[fit])])
    both = np.concatenate([fit, tune])
    m = LogisticRegression(C=best[1], max_iter=300).fit(Z(both), y[both])
    ll = nll(m, test)
    out = np.full(n_subj, np.nan)
    c = np.bincount(subj[test], minlength=n_subj)
    out[c > 0] = np.bincount(subj[test], ll, minlength=n_subj)[c > 0] / c[c > 0]
    return out


def tune_scores(F, an):
    """Mean log loss on the tune subjects of a probe fitted on the fit subjects (C = 0.1): the number that picks a pilot's settings."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    out = {}
    for key, y in (("tune_detect", an["detect"].astype(np.int64)), ("tune_stage", an["stage"].astype(np.int64))):
        ok = y >= 0
        fit = rng.permutation(np.flatnonzero(ok & (an["split"] == 0)))[:50_000]
        tune = rng.permutation(np.flatnonzero(ok & (an["split"] == 1)))[:50_000]
        mu, sd = F[fit].mean(0), F[fit].std(0) + 1e-6
        m = LogisticRegression(C=0.1, max_iter=300).fit((F[fit] - mu) / sd, y[fit])
        p = m.predict_proba((F[tune] - mu) / sd)[np.arange(len(tune)), np.searchsorted(m.classes_, y[tune])]
        out[key] = float(-np.log(np.clip(p, 1e-6, 1)).mean())
    return out


def cmd_probe(a):
    an = np.load(Path(a.work) / "anchors.npz")
    n = len(an["subjects"])
    F = np.load(a.emb).astype(np.float32)
    det = pa._lr_loss(F, an["detect"], an["split"], an["subj"], n)
    stage = stage_losses(F, an["stage"].astype(np.int64), an["split"], an["subj"], n)
    np.savez(a.out, detect=det, stage=stage, **tune_scores(F, an))
    print("probed", a.out, "detect", float(np.nanmean(det)), "stage", float(np.nanmean(stage)), flush=True)


def base_rates(an):
    """Per-subject test log loss of the fit-set base rate for detect and of the fit-set class frequencies for stage."""
    n = len(an["subjects"])
    fit, test = an["split"] == 0, an["split"] == 2
    ok = an["detect"] >= 0
    p = np.clip(an["detect"][fit & ok].mean(), 1e-6, 1 - 1e-6)
    d_ll = np.full(n, np.nan)
    t = test & ok
    ll = -(an["detect"][t] * np.log(p) + (1 - an["detect"][t]) * np.log(1 - p))
    c = np.bincount(an["subj"][t], minlength=n)
    d_ll[c > 0] = np.bincount(an["subj"][t], ll, minlength=n)[c > 0] / c[c > 0]
    freq = np.clip(np.bincount(an["stage"][fit], minlength=5) / fit.sum(), 1e-6, 1)
    ll = -np.log(freq[an["stage"][test]])
    s_ll = np.full(n, np.nan)
    c = np.bincount(an["subj"][test], minlength=n)
    s_ll[c > 0] = np.bincount(an["subj"][test], ll, minlength=n)[c > 0] / c[c > 0]
    return d_ll, s_ll


def cmd_g0(a):
    work = Path(a.work)
    an = np.load(work / "anchors.npz")
    test_subj = sorted(set(an["subj"][an["split"] == 2].tolist()))
    rng = np.random.default_rng(0)
    files = {"pretrained": a.ckpt, "random0": 0, "random1": 1, "random2": 2, "bandpower": None}
    loss = {}
    for name, what in files.items():
        emb, pr = work / f"g0-{name}.npy", work / f"g0-{name}.npz"
        if not pr.exists():
            if not emb.exists():
                cmd_embed(_Args(shards=a.shards, work=a.work, workers=a.workers, out=str(emb), bandpower=name == "bandpower",
                                random=what if name.startswith("random") else None, like=a.ckpt, ckpt=a.ckpt))
            cmd_probe(_Args(work=a.work, emb=str(emb), out=str(pr)))
        loss[name] = np.load(pr)
    d_base, s_base = base_rates(an)
    verdict = {}
    for readout in ("detect", "stage"):
        rnd = np.mean([loss[f"random{s}"][readout] for s in range(3)], 0)
        gain = (rnd - loss["pretrained"][readout]) / LN2
        m, lo, hi, p, k = pa.boot(gain[test_subj], rng)
        band = (loss["bandpower"][readout] - loss["pretrained"][readout]) / LN2
        bm = pa.boot(band[test_subj], rng)
        base = (d_base if readout == "detect" else s_base)
        verdict[readout] = {"bits_pretrained_over_random": [m, lo, hi], "pass": bool(lo > 0), "subjects": k,
                            "bits_pretrained_over_bandpower": list(bm[:3]),
                            "bits_pretrained_over_base_rate": float(np.nanmean((base - loss["pretrained"][readout])[test_subj]) / LN2)}
    primary = "detect" if verdict["detect"]["pass"] else ("stage" if verdict["stage"]["pass"] else None)
    verdict["primary"] = primary
    (work / "g0.json").write_text(json.dumps(verdict, indent=1))
    print(json.dumps(verdict, indent=1))
    print(f"G0_VERDICT primary={primary}" if primary else "G0_VERDICT E1_STOPS: neither detect nor stage beats random weights")


def cmd_g0tune(a):
    """Add the tune scores to the G0 random-weights probes (they were written before tune_scores existed)."""
    work = Path(a.work)
    an = np.load(work / "anchors.npz")
    for s in range(3):
        f = work / f"g0-random{s}.npz"
        old = dict(np.load(f))
        if "tune_detect" not in old:
            np.savez(f, **old, **tune_scores(np.load(work / f"g0-random{s}.npy").astype(np.float32), an))
    print("g0tune done")


RUNS = "{arm}-{size}-{cell}-s{seed}"
ARMS, SIZES, SEEDS = ("signal", "latent"), ("10m", "50m", "200m"), (0, 1, 2)


def planned(rung2_cells=None, seeds_200m=3, arms=ARMS):
    """The grid: 5 data cells x 3 sizes x 2 arms x 3 seeds, minus what K1 (a failed latent arm), K2 and K3 cut at 200m."""
    cells = ("r1", "r4", "r16", "clean4", "dirty4")
    out = []
    for size in SIZES:
        cs = cells if size != "200m" or rung2_cells is None else rung2_cells
        seeds = SEEDS if size != "200m" else SEEDS[:seeds_200m]
        out += [RUNS.format(arm=a, size=size, cell=c, seed=s) for a in arms for c in cs for s in seeds]
    return out


def cmd_report(a):
    work = Path(a.work)
    cuts = a.cells_200m.split(",") if a.cells_200m else None
    frozen = json.loads((work / "frozen.json").read_text()) if (work / "frozen.json").exists() else {}
    arms = tuple(frozen.get("arms", ARMS))
    missing = [r for r in planned(cuts, a.seeds_200m, arms) if not (work / "runs" / r / "probe.npz").exists()]
    if missing:
        raise SystemExit(f"refusing to report: {len(missing)} planned runs have no probe.npz, first {missing[:3]}")
    an = np.load(work / "anchors.npz")
    test_subj = sorted(set(an["subj"][an["split"] == 2].tolist()))
    rng = np.random.default_rng(0)
    g0 = json.loads((work / "g0.json").read_text())
    readout = g0["primary"]
    from kahlus.model import SIZES as SPEC
    from kahlus.model import KahlusPatchModel, ModelConfig, n_params

    nparams = {s: n_params(KahlusPatchModel(ModelConfig(**SPEC[s], mlp_head=True))) for s in SIZES}
    x = np.log10([nparams[s] for s in SIZES])

    def loss(arm, size, cell, key):
        return np.mean([np.load(work / "runs" / RUNS.format(arm=arm, size=size, cell=cell, seed=s) / "probe.npz")[key] for s in SEEDS], 0)

    def slope(arm, cell, key, seeds=SEEDS):
        ys = np.stack([np.mean([np.load(work / "runs" / RUNS.format(arm=arm, size=s_, cell=cell, seed=sd) / "probe.npz")[key] for sd in seeds], 0)
                       for s_ in SIZES])  # (sizes, subjects)
        out = np.full(ys.shape[1], np.nan)
        ok = np.isfinite(ys).all(0)
        out[ok] = np.polyfit(x, ys[:, ok], 1)[0]  # one line per subject over the 3 sizes
        return out

    lines = [f"# E1 report (primary readout: {readout})\n", "| Arm | Size | Cell | Mean test log loss (bits) |", "|---|---|---|---|"]
    for arm in ARMS:
        for size in SIZES:
            for cell in ("r1", "r4", "r16", "clean4", "dirty4"):
                if (work / "runs" / RUNS.format(arm=arm, size=size, cell=cell, seed=0) / "probe.npz").exists():
                    lines.append(f"| {arm} | {size} | {cell} | {np.nanmean(loss(arm, size, cell, readout)[test_subj]) / LN2:.4f} |")
    if "latent" not in arms:  # K1: the latent arm failed at 10M, so there is no contrast to test
        (work / "report.md").write_text("\n".join(lines) + "\n\nThe latent arm collapsed at 10M (K1): the grid ran the signal arm only and there is no D.\n")
        print("\n".join(lines))
        return
    d = slope("latent", "r1", readout) - slope("signal", "r1", readout)
    m, lo, hi, _, k = pa.boot(d[test_subj], rng)
    per_seed = [float(np.nanmean((slope("latent", "r1", readout, (s,)) - slope("signal", "r1", readout, (s,)))[test_subj])) for s in SEEDS]
    lines += [f"\n**Primary D** (latent slope minus signal slope of log loss per decade of parameters, r1): {m:+.4f} [{lo:+.4f}, {hi:+.4f}] nats "
              f"over {k} subjects; per seed {per_seed}. Positive D: the signal target gains more from size."]
    cells = {}
    both = lambda size, cell, key=readout: np.mean([loss(arm, size, cell, key) for arm in ARMS], 0)  # noqa: E731  mean over the two arms
    other = "stage" if readout == "detect" else "detect"
    hours = np.array([-4.0, -2.0, 0.0])  # log2 of the hours of r16, r4, r1 relative to r1 (the cells hold 1/16, 1/4, all nights)
    hslope = lambda: -np.array([np.polyfit(hours, np.stack([both("50m", c)[i] for c in ("r16", "r4", "r1")]), 1)[0] if np.isfinite(  # noqa: E731
        [both("50m", c)[i] for c in ("r16", "r4", "r1")]).all() else np.nan for i in range(len(an["subjects"]))])
    d2 = slope("latent", "r1", other) - slope("signal", "r1", other)
    for name, f in (("clean4 minus r4 (mean of arms, 50m)", lambda: both("50m", "r4") - both("50m", "clean4")),
                    ("clean4 minus r1 (mean of arms, 50m)", lambda: both("50m", "r1") - both("50m", "clean4")),
                    ("loss gain per doubling of hours, r16 to r1 (mean of arms, 50m)", hslope),
                    ("size slope, signal arm (r1)", lambda: -slope("signal", "r1", readout)),
                    ("size slope, latent arm (r1)", lambda: -slope("latent", "r1", readout)),
                    (f"primary D on the {other} readout", lambda: d2)):
        cells[name] = pa.boot(f()[test_subj], rng)
    adj = pa.holm([v[3] for v in cells.values()])
    lines += ["\n| Secondary cell (positive favours the first named) | Mean nats | 95% interval | Holm p |", "|---|---|---|---|"]
    for (name, v), p in zip(cells.items(), adj):
        lines.append(f"| {name} | {v[0]:+.4f} | [{v[1]:+.4f}, {v[2]:+.4f}] | {p:.3f} |")
    out = work / "report.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("anchors", "embed", "probe", "g0", "g0tune", "report"))
    ap.add_argument("--shards", default="/shared/aayu/fill-a/shards/shhs1-full")
    ap.add_argument("--nsrr", default="/shared/aayu/nsrr/shhs/polysomnography")
    ap.add_argument("--work", default="/shared/aayu/e1")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ckpt")
    ap.add_argument("--like", help="a checkpoint whose model config random weights copy")
    ap.add_argument("--random", type=int)
    ap.add_argument("--bandpower", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--emb")
    ap.add_argument("--cells-200m", help="comma list of 200m cells K2 or K3 left in the plan")
    ap.add_argument("--seeds-200m", type=int, default=3, help="seeds per 200m cell K3 left in the plan")
    a = ap.parse_args()
    {"anchors": cmd_anchors, "embed": cmd_embed, "probe": cmd_probe, "g0": cmd_g0, "g0tune": cmd_g0tune, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
