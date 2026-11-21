"""Pilots A and B on the partial SHHS-1 shards (exploratory, not a claim). Specs: docs/prereg/pilot_a.md, pilot_b.md.

  anchors   (CPU)  arousal anchors and 64 s window positions per night, plus the Pilot B epoch windows -> anchors.npz, bwin.npz
  embed     (GPU)  one embedding file per (init, seed, input, set)
  probe     (CPU)  linear probes for Pilot A at 30 s, and the prior-only logistic at every horizon
  finetune  (GPU)  encoder + linear head on [embedding, prior], one run per (init, seed, lr, shift)
  probe-b   (CPU)  Pilot B fingerprint probes
  report    (CPU)  per-subject bits, paired subject bootstrap, Holm -> report.md, per_subject.csv

Every window ends at least 2 s before its anchor, so no look-ahead (asserted in `window_end`). Stored patch j of a night
covers second [j + WARMUP, j + WARMUP + 1) of the recording (shards.causal_patches drops 60 patches and the first is a
warm-up for the running z-score).
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path

import numpy as np

from kahlus import heart_arousal as ha
from kahlus.forecast_bench import is_tuning
from kahlus.nsrr import event_kind, parse_nsrr_events, sleep_stage_epochs, stage_at
from kahlus.shards import FS_HZ, WARMUP_PATCHES, ShardSet, is_heldout

CTX = 64  # seconds in a window
MARGIN_S = 2  # a window ends at least this long before its anchor
DEV_IDS = range(200001, 200026)  # the dev nights are never used
HOR = ha.HORIZONS_S
H30 = HOR.index(30)
STAGES = ("Wake", "Stage 1 sleep", "Stage 2 sleep", "Stage 3 sleep", "REM sleep")  # Stage 4 joins Stage 3
LRS = (1e-4, 3e-4, 1e-3)
FT_STEPS, FT_BATCH, FT_WARMUP = 4000, 256, 200  # fixed by the preregistration; --steps only shortens a smoke run
N_BOOT = 2000


# ---------------------------------------------------------------- windows and anchors
def window_end(t: float) -> int:
    """Index of the last stored patch of the window for an anchor (or epoch boundary) at ``t`` seconds."""
    j = int(math.floor(t)) - WARMUP_PATCHES - 1 - MARGIN_S
    assert j + WARMUP_PATCHES + 1 <= t - MARGIN_S  # the window's last second ends at least MARGIN_S before t
    return j


def subject_of(night: dict) -> int:
    return int(re.sub(r"\D", "", night["subject"]))


def split_of(night: dict) -> int:  # 0 fit, 1 tune, 2 test
    return 2 if is_heldout("shhs", night["subject"]) else 1 if is_tuning(night) else 0


def b_subjects(nights: list[dict]) -> list[int]:
    """The first 200 held-out subjects by ID: Pilot B's subjects."""
    return sorted(subject_of(n) for n in nights if split_of(n) == 2 and subject_of(n) not in DEV_IDS)[:200]


def one_night(job):
    night, xml, edf, want_b = job
    n = night["n"]
    events = parse_nsrr_events(xml)
    t, lab = ha.anchor_labels(events, total_s=n + WARMUP_PATCHES)
    j = np.asarray([window_end(x) for x in t], dtype=np.int64) if t.size else np.zeros(0, np.int64)
    ok = (j >= CTX - 1) & (j < n)
    out = {"a": (t[ok].astype(np.float32), j[ok].astype(np.int32), lab[ok], ha.prior_features(events, t[ok]))}
    out["onsets"] = np.asarray([s for c, s, _ in events if event_kind(c) == "arousal"])
    if want_b:
        stages = sleep_stage_epochs(events)
        bt = np.arange(90.0, n + WARMUP_PATCHES, 30.0)
        bj = np.asarray([window_end(x) for x in bt], dtype=np.int64)
        st = [stage_at(stages, x - 15.0) for x in bt]
        st = [STAGES.index("Stage 3 sleep" if s == "Stage 4 sleep" else s) if s in STAGES or s == "Stage 4 sleep" else -1 for s in st]
        keep = (bj >= CTX - 1) & (bj < n) & (np.asarray(st) >= 0)
        from kahlus.nsrr import load_nsrr_epoch_matrix

        m = load_nsrr_epoch_matrix(edf, channel_groups={"eeg": ("EEG",)})
        eeg, fs = m["epochs"]["eeg"][:, :, 0].ravel(), m["sfreq"]
        rms = np.asarray([np.sqrt(np.mean(eeg[int((x + WARMUP_PATCHES + 1 - CTX) * fs): int((x + WARMUP_PATCHES + 1) * fs)] ** 2)) * 1e6
                          for x in bj[keep]], dtype=np.float32)
        out["b"] = (bt[keep].astype(np.float32), bj[keep].astype(np.int32), np.asarray(st, np.int8)[keep], rms)
    return out


def alignment_ratio(shards: ShardSet, nights: list[dict], onsets: list[np.ndarray]) -> float:
    """Mean C4 power in the 5 s after an arousal onset over the 5 s before it (> 1.2 means the stored time axis is right)."""
    r = []
    for night, on in zip(nights, onsets):
        for s in on[:30]:
            a = int(s) - WARMUP_PATCHES
            if a - 5 < 0 or a + 5 > night["n"]:
                continue
            x = np.asarray(shards.read(night, a - 5, 10), dtype=np.float32)[:, 0] ** 2
            r.append(x[5:].mean() / max(x[:5].mean(), 1e-6))
    return float(np.mean(r)) if r else float("nan")


def cmd_anchors(a):
    from multiprocessing import Pool

    shards = ShardSet(a.shards)
    nights = [n for n in shards.nights if n["dataset"] == "shhs" and subject_of(n) not in DEV_IDS]
    bset = set(b_subjects(nights))
    xml_dir, edf_dir = Path(a.nsrr) / "annotations-events-nsrr/shhs1", Path(a.nsrr) / "edfs/shhs1"
    jobs, kept = [], []
    for n in nights:
        x = xml_dir / f"shhs1-{n['subject']}-nsrr.xml"
        if x.exists():
            jobs.append((n, x, edf_dir / f"shhs1-{n['subject']}.edf", subject_of(n) in bset))
            kept.append(n)
    print(f"shard nights {len(nights)}, with an XML on the mirror {len(kept)}", flush=True)
    with Pool(max(1, a.workers)) as pool:
        res = list(pool.imap(one_night, jobs, chunksize=4))
    A = {k: [] for k in ("subj", "t", "j", "lab", "prior", "split")}
    B = {k: [] for k in ("subj", "t", "j", "stage", "rms")}
    for i, (n, r) in enumerate(zip(kept, res)):
        t, j, lab, prior = r["a"]
        for k, v in zip(A, (np.full(t.size, i, np.int32), t, j, lab, prior, np.full(t.size, split_of(n), np.int8))):
            A[k].append(v)
        if "b" in r:
            bt, bj, st, rms = r["b"]
            for k, v in zip(B, (np.full(bt.size, i, np.int32), bt, bj, st, rms)):
                B[k].append(v)
    A = {k: np.concatenate(v) for k, v in A.items()}
    B = {k: np.concatenate(v) for k, v in B.items()}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    subj = np.asarray([n["subject"] for n in kept])
    np.savez(out / "anchors.npz", subjects=subj, **A)
    np.savez(out / "bwin.npz", subjects=subj, **B)
    test = A["split"] == 2
    print(f"subjects: {len(kept)} (fit {sum(split_of(n) == 0 for n in kept)}, tune {sum(split_of(n) == 1 for n in kept)}, "
          f"test {sum(split_of(n) == 2 for n in kept)}); anchors {len(A['t'])}; test anchors {int(test.sum())}", flush=True)
    for k, h in enumerate(HOR):
        print(f"  horizon {h:3d}s: positives fit {int((A['lab'][A['split'] == 0, k] == 1).sum())}, test {int((A['lab'][test, k] == 1).sum())}", flush=True)
    print(f"test subjects with a 30 s label: {len(set(A['subj'][test & (A['lab'][:, H30] >= 0)]))}; Pilot B windows {len(B['t'])}", flush=True)
    tests = [i for i, n in enumerate(kept) if split_of(n) == 2][:20]
    print("ALIGNMENT_RATIO", round(alignment_ratio(shards, [kept[i] for i in tests], [res[i]["onsets"] for i in tests]), 3), flush=True)


def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- windows to tensors
class Windows:
    """Window (64, 2, 125) float32 per row of an index set. ``shift`` takes the window from half a night later."""

    def __init__(self, shards: ShardSet, nights, subj, j, *, perwindow=False, shift=False):
        self.shards, self.nights, self.subj, self.j, self.perwindow, self.shift = shards, nights, subj, j, perwindow, shift

    def __len__(self):
        return len(self.j)

    def __getitem__(self, i):
        n = self.nights[self.subj[i]]
        j = int(self.j[i])
        if self.shift:
            j = max((j + n["n"] // 2) % n["n"], CTX - 1)
        x = np.asarray(self.shards.read(n, j - CTX + 1, CTX), dtype=np.float32)
        if self.perwindow:
            x = (x - x.mean(0)) / np.maximum(x.std(0), 1e-6)
        return x, i


def load(args):
    shards = ShardSet(args.shards)
    a = np.load(Path(args.work) / "anchors.npz")
    by_subject = {n["subject"]: n for n in shards.nights}
    nights = [by_subject[s] for s in a["subjects"]]
    return shards, nights, a


def make_model(init: str, seed: int, ckpt: str):
    import torch

    from kahlus.model import KahlusPatchModel, ModelConfig

    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    torch.manual_seed(seed)
    model = KahlusPatchModel(ModelConfig(**state["identity"]["model"]))
    if init == "pretrained":
        model.load_state_dict(state["model"])
    return model


def band_features(x: np.ndarray) -> np.ndarray:
    """Relative band power in 5 bands and log power of channel 0 (C4) of each window (N, 64, 2, 125)."""
    c4 = x[:, :, 0].reshape(len(x), -1)
    p = np.abs(np.fft.rfft(c4, axis=1)) ** 2
    f = np.fft.rfftfreq(c4.shape[1], 1 / FS_HZ)
    bands = [p[:, (f >= lo) & (f < hi)].sum(1) for lo, hi in ha.EEG_BANDS_HZ.values()]
    tot = np.maximum(sum(bands), 1e-9)
    return np.stack([b / tot for b in bands] + [np.log(tot)], 1).astype(np.float32)


def cmd_embed(a):
    import torch

    shards, nights, an = load(a)
    z = np.load(Path(a.work) / ("anchors.npz" if a.set == "a" else "bwin.npz"))
    ds = Windows(shards, nights, z["subj"], z["j"], perwindow=a.input == "perwindow")
    dl = torch.utils.data.DataLoader(ds, batch_size=1024, num_workers=a.workers)
    out = []
    if a.init == "bandpower":
        out = [band_features(x.numpy()) for x, _ in dl]
    else:
        model = make_model(a.init, a.seed, a.ckpt).to(device()).eval()
        with torch.no_grad(), torch.autocast(device(), dtype=torch.bfloat16):
            out = [model.encode(x.to(device())).float().flatten(1).cpu().numpy().astype(np.float16) for x, _ in dl]
    np.save(a.out, np.concatenate(out))
    print("embedded", a.out, flush=True)


# ---------------------------------------------------------------- finetune
def log_loss_by_subject(p, y, subj, n_subj):
    """Mean log loss per subject, per horizon: (n_subj, 5); NaN where a subject has no labelled anchor."""
    out = np.full((n_subj, y.shape[1]), np.nan)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    for k in range(y.shape[1]):
        ok = y[:, k] >= 0
        ll = -(y[ok, k] * np.log(p[ok, k]) + (1 - y[ok, k]) * np.log(1 - p[ok, k]))
        s = np.bincount(subj[ok], ll, minlength=n_subj)
        c = np.bincount(subj[ok], minlength=n_subj)
        out[c > 0, k] = s[c > 0] / c[c > 0]
    return out


def cmd_finetune(a):
    import torch

    shards, nights, an = load(a)
    out = Path(a.out)
    if (out / "pred.npz").exists():
        print("skip", out)
        return
    out.mkdir(parents=True, exist_ok=True)
    fit_rows = an["split"] == 0
    mu, sd = an["prior"][fit_rows].mean(0), an["prior"][fit_rows].std(0) + 1e-6
    prior = torch.from_numpy(np.clip((an["prior"] - mu) / sd, -10, 10).astype(np.float32))
    lab = torch.from_numpy(an["lab"].astype(np.float32))
    ds = Windows(shards, nights, an["subj"], an["j"], shift=bool(a.shift))
    fit = np.flatnonzero(fit_rows)
    draw = fit[np.random.default_rng(a.seed).integers(0, len(fit), FT_STEPS * FT_BATCH)]  # the same fit rows for both arms of a seed
    dl = torch.utils.data.DataLoader(ds, batch_size=FT_BATCH, sampler=draw.tolist(), num_workers=a.workers, drop_last=True)
    enc = make_model(a.init, a.seed, a.ckpt).to(device())
    head = torch.nn.Linear(enc.cfg.d_model * enc.cfg.n_channels + prior.shape[1], len(HOR)).to(device())
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / FT_WARMUP) * 0.5 * (1 + math.cos(math.pi * s / FT_STEPS)))

    def forward(x, rows):
        with torch.autocast(device(), dtype=torch.bfloat16):
            h = enc.encode(x.to(device())).flatten(1)
        return head(torch.cat([h.float(), prior[rows].to(device())], 1))

    enc.train()
    for step, (x, rows) in enumerate(dl):
        y = lab[rows].to(device())
        ok = (y >= 0).float()
        bce = torch.nn.functional.binary_cross_entropy_with_logits(forward(x, rows), y.clamp(min=0), reduction="none")
        loss = (bce * ok).sum() / ok.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 200 == 0:
            print(f"step {step} loss {loss.item():.4f}", flush=True)
    enc.eval()
    preds = {}
    for name, sp in (("tune", 1), ("test", 2)):
        idx = np.flatnonzero(an["split"] == sp)
        ps = []
        with torch.no_grad():
            for x, rows in torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=1024, num_workers=a.workers):
                ps.append(torch.sigmoid(forward(x, rows)).cpu().numpy())
        preds[name] = np.concatenate(ps)
    tune = np.flatnonzero(an["split"] == 1)
    ll = log_loss_by_subject(preds["tune"], an["lab"][tune], an["subj"][tune], len(an["subjects"]))
    np.savez(out / "pred.npz", tune=preds["tune"], test=preds["test"], tune_ll30=float(np.nanmean(ll[:, H30])))
    print("tune log loss at 30 s", float(np.nanmean(ll[:, H30])), flush=True)


# ---------------------------------------------------------------- probes
def _lr_loss(F, y, split, subj, n_subj, seed=0):
    """Logistic regression on F for one horizon: C from {0.01, 0.1, 1} by tune log loss, refit on fit + tune, per-subject test loss."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(seed)
    ok = y >= 0
    pick = lambda sp, cap: (lambda r: rng.permutation(r)[:cap])(np.flatnonzero(ok & (split == sp)))  # noqa: E731
    fit, tune = pick(0, 100_000), pick(1, 100_000)
    test = np.flatnonzero(ok & (split == 2))
    mu, sd = F[fit].mean(0), F[fit].std(0) + 1e-6
    Z = lambda r: ((F[r] - mu) / sd).astype(np.float32)  # noqa: E731
    best = min((float(np.mean(-np.log(np.where(y[tune] == 1, p[:, 1], p[:, 0]) + 1e-9))), c)
               for c in (0.01, 0.1, 1.0) for p in [LogisticRegression(C=c, max_iter=2000).fit(Z(fit), y[fit]).predict_proba(Z(tune))])
    m = LogisticRegression(C=best[1], max_iter=2000).fit(Z(np.concatenate([fit, tune])), y[np.concatenate([fit, tune])])
    p = np.clip(m.predict_proba(Z(test))[:, 1], 1e-6, 1 - 1e-6)
    ll = -(y[test] * np.log(p) + (1 - y[test]) * np.log(1 - p))
    out = np.full(n_subj, np.nan)
    c = np.bincount(subj[test], minlength=n_subj)
    out[c > 0] = np.bincount(subj[test], ll, minlength=n_subj)[c > 0] / c[c > 0]
    return out


def cmd_probe(a):
    from joblib import Parallel, delayed

    an = np.load(Path(a.work) / "anchors.npz")
    n_subj = len(an["subjects"])
    jobs = {f"prior_h{h}": (an["prior"], k) for k, h in enumerate(HOR)}
    for name in ("bandpower", "emb-pretrained-asis-s0", "emb-pretrained-perwindow-s0", "emb-random-asis-s0", "emb-random-asis-s1", "emb-random-asis-s2"):
        f = Path(a.work) / f"{name.replace('emb-', 'emb-a-')}.npy" if name != "bandpower" else Path(a.work) / "bandpower-a-asis-s0.npy"
        if f.exists():
            jobs[name + "_h30"] = (np.concatenate([an["prior"], np.load(f).astype(np.float32)], 1), H30)
    res = Parallel(n_jobs=a.jobs)(delayed(_lr_loss)(F, an["lab"][:, k].astype(np.int8), an["split"], an["subj"], n_subj) for F, k in jobs.values())
    np.savez(Path(a.work) / "probe.npz", **dict(zip(jobs, res)))
    print("probe sets:", list(jobs))


def _fit_predict(Ftr, ytr, Fte, c=1.0):
    from sklearn.linear_model import LogisticRegression

    if len(np.unique(ytr)) < 2:  # a one-class training set (only in tiny smoke data): predict that class
        return np.full(len(Fte), ytr[0])
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    return LogisticRegression(C=c, max_iter=200).fit((Ftr - mu) / sd, ytr).predict((Fte - mu) / sd)


def _probe_b(F, bw):
    """Subject-ID, amplitude and stage probes on one feature set; returns what the report bootstraps."""
    subj = bw["subj"]
    ids = np.unique(subj)
    rank = np.zeros(len(subj))  # position of a window inside its night, 0..1
    for s in ids:
        r = np.flatnonzero(subj == s)
        rank[r[np.argsort(bw["j"][r])]] = np.arange(len(r)) / len(r)
    train, test = np.flatnonzero(rank < 0.4), np.flatnonzero(rank >= 0.6)
    train = train[::max(1, len(train) // (100 * len(ids)))]
    pid = _fit_predict(F[train], subj[train], F[test])
    out = {"id_subjects": ids, "id_acc": np.asarray([np.mean(pid[subj[test] == s] == s) for s in ids])}
    cut = int(len(ids) * 0.8)
    tr_s, te_s = ids[:cut], ids[cut:]  # 160 / 40 subjects of 200: a different subject split for amplitude and stage
    tr, te = np.flatnonzero(np.isin(subj, tr_s)), np.flatnonzero(np.isin(subj, te_s))
    edges = np.quantile(bw["rms"], [0.2, 0.4, 0.6, 0.8])
    amp = np.digitize(bw["rms"], edges)
    pa = _fit_predict(F[tr[::5]], amp[tr[::5]], F[te])
    ps = _fit_predict(F[tr[::5]], bw["stage"][tr[::5]], F[te])
    from sklearn.metrics import balanced_accuracy_score

    out |= {"amp_acc": float(np.mean(pa == amp[te])), "stage_ba": float(balanced_accuracy_score(bw["stage"][te], ps)),
            "stage_true": bw["stage"][te], "stage_pred": ps, "stage_subj": subj[te], "amp_true": amp[te], "amp_pred": pa}
    return out


def cmd_probe_b(a):
    from joblib import Parallel, delayed

    bw = dict(np.load(Path(a.work) / "bwin.npz"))  # a plain dict: an NpzFile cannot go to joblib workers
    sets = {}
    for f in sorted(Path(a.work).glob("emb-b-*.npy")) + sorted(Path(a.work).glob("bandpower-b-*.npy")):
        sets[f.stem] = np.load(f).astype(np.float32)
    res = Parallel(n_jobs=a.jobs)(delayed(_probe_b)(F, bw) for F in sets.values())
    np.save(Path(a.work) / "probe_b.npy", np.asarray(dict(zip(sets, res)), dtype=object), allow_pickle=True)
    for k, r in zip(sets, res):
        print(k, "id", round(float(r["id_acc"].mean()), 4), "amp", round(r["amp_acc"], 4), "stage BA", round(r["stage_ba"], 4))


# ---------------------------------------------------------------- report
def boot(vals: np.ndarray, rng: np.random.Generator):
    """Mean, 95% percentile interval and two-sided p of the mean of per-subject values (paired subject bootstrap)."""
    v = vals[np.isfinite(vals)]
    m = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(N_BOOT)])
    p = 2 * min((np.sum(m <= 0) + 1) / (N_BOOT + 1), (np.sum(m >= 0) + 1) / (N_BOOT + 1))
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), min(1.0, p), len(v)


def holm(ps):
    order = np.argsort(ps)
    adj, run = np.zeros(len(ps)), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def pick_runs(work: Path, init: str, seed: int):
    runs = [(float(np.load(d / "pred.npz")["tune_ll30"]), d) for d in work.glob(f"ft-{init}-s{seed}-lr*") if (d / "pred.npz").exists()]
    return min(runs)[1] if runs else None


def cmd_report(a):
    work = Path(a.work)
    an = np.load(work / "anchors.npz")
    n_subj = len(an["subjects"])
    test_rows = np.flatnonzero(an["split"] == 2)
    test_subj = sorted(set(an["subj"][test_rows].tolist()))
    rng = np.random.default_rng(0)
    probe = np.load(work / "probe.npz")
    ln2 = math.log(2)
    arm = {}
    for init in ("pretrained", "random"):
        ls = []
        for seed in (0, 1, 2):
            d = pick_runs(work, init, seed)
            if d:
                ls.append(log_loss_by_subject(np.load(d / "pred.npz")["test"], an["lab"][test_rows], an["subj"][test_rows], n_subj))
        arm[init] = np.mean(ls, 0) if ls else None
    prior = np.stack([probe[f"prior_h{h}"] for h in HOR], 1)
    cells = []  # (name, per-subject values in bits, positive = the first beats the second)
    for k, h in enumerate(HOR):
        if arm["pretrained"] is not None and arm["random"] is not None:
            cells.append((f"fine-tune pretrained minus scratch, {h} s", (arm["random"][:, k] - arm["pretrained"][:, k]) / ln2))
    get = lambda n: probe[n] if n in probe else None  # noqa: E731
    rnd = [get(f"emb-random-asis-s{s}_h30") for s in (0, 1, 2) if get(f"emb-random-asis-s{s}_h30") is not None]
    pre, pw, bp = get("emb-pretrained-asis-s0_h30"), get("emb-pretrained-perwindow-s0_h30"), get("bandpower_h30")
    if pre is not None and rnd:
        cells.append(("probe pretrained minus random-init, 30 s", (np.mean(rnd, 0) - pre) / ln2))
    if pre is not None and bp is not None:
        cells.append(("probe pretrained minus prior + band power, 30 s", (bp - pre) / ln2))
    if pre is not None and pw is not None:
        cells.append(("probe per-window input minus as-is input, 30 s", (pre - pw) / ln2))
    primary = next((c for c in cells if c[0].endswith("scratch, 30 s")), None)
    secondary = [c for c in cells if c is not primary]
    stats = {n: boot(v[test_subj], rng) for n, v in cells}
    adj = dict(zip([n for n, _ in secondary], holm([stats[n][3] for n, _ in secondary]))) if secondary else {}
    lines = ["# Pilots A and B on partial SHHS-1 shards (exploratory, not a claim)\n",
             f"Test subjects: {len(test_subj)} (held out of pretraining). Bits = log-loss difference / ln 2, positive favours the first arm. "
             f"Paired subject bootstrap, {N_BOOT} resamples, seed 0.\n", "## Pilot A\n", "| Cell | Mean bits | 95% interval | Subjects | Holm p |", "|---|---|---|---|---|"]
    for n, _ in ([primary] if primary else []) + secondary:
        m, lo, hi, p, k = stats[n]
        lines.append(f"| {n}{' (primary)' if n == (primary or ('',))[0] else ''} | {m:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {k} | {adj.get(n, float('nan')):.3f} |")
    # shift control: the pretrained fine-tune on windows from half a night away, against the prior-only logistic at 30 s
    ctl = None
    shift_ll = []
    for seed in (0, 1, 2):
        d = work / f"ft-pretrained-s{seed}-shift"
        if (d / "pred.npz").exists():
            shift_ll.append(log_loss_by_subject(np.load(d / "pred.npz")["test"], an["lab"][test_rows], an["subj"][test_rows], n_subj))
    if shift_ll:
        ctl = boot(((prior[:, H30] - np.mean(shift_ll, 0)[:, H30]) / ln2)[test_subj], rng)
        lines.append(f"\nShift control (pretrained fine-tune on half-night-shifted windows, bits over the prior-only logistic, 30 s): "
                     f"{ctl[0]:+.4f} [{ctl[1]:+.4f}, {ctl[2]:+.4f}]")
    for init in ("pretrained", "random"):
        if arm[init] is not None:
            m = boot(((prior[:, H30] - arm[init][:, H30]) / ln2)[test_subj], rng)
            lines.append(f"\nFine-tune {init}, bits over the prior-only logistic at 30 s: {m[0]:+.4f} [{m[1]:+.4f}, {m[2]:+.4f}]")
    ctl_ok = ctl is not None and ctl[1] <= 0 <= ctl[2]
    if primary is None or ctl is None:
        verdict = "not reported: the fine-tune or shift-control runs are missing"
    elif not ctl_ok:
        verdict = "NOT REPORTED: shift control failed"
    else:
        m, lo, hi, _, _ = stats[primary[0]]
        verdict = "clear gain" if lo > 0 and m >= 0.002 else "no clear gain"
    lines += [f"\n**Pilot A verdict: {verdict}.** Gate for larger sizes: the primary interval above 0, mean at least +0.002 bits, shift control passes."]
    lines += ["\n## Pilot B\n"] + report_b(work, rng)
    (work / "report.md").write_text("\n".join(lines) + "\n")
    with open(work / "per_subject.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "prior_loss_30s", "ft_pretrained_30s", "ft_random_30s"])
        for s in test_subj:
            w.writerow([an["subjects"][s], prior[s, H30], *(arm[i][s, H30] if arm[i] is not None else "" for i in ("pretrained", "random"))])
    print("\n".join(lines))
    primary_txt = f"{stats[primary[0]][0]:+.4f} [{stats[primary[0]][1]:+.4f}, {stats[primary[0]][2]:+.4f}]" if primary else "n/a"
    print(f"PILOT_A_VERDICT {verdict}; primary {primary_txt}")


def report_b(work: Path, rng) -> list[str]:
    f = work / "probe_b.npy"
    if not f.exists():
        return ["Pilot B probes missing."]
    pb = np.load(f, allow_pickle=True).item()
    rows = ["| Feature set | Subject-ID acc (chance 0.005) | Amplitude acc (chance 0.2) | Stage balanced acc (chance 0.2) |", "|---|---|---|---|"]
    for k, r in pb.items():
        rows.append(f"| {k} | {r['id_acc'].mean():.4f} | {r['amp_acc']:.4f} | {r['stage_ba']:.4f} |")
    g = lambda init, inp: [pb[k] for k in pb if k.startswith(f"emb-b-{init}-{inp}")]  # noqa: E731
    out = rows
    for inp in ("asis",):
        pre, rnd = g("pretrained", inp), g("random", inp)
        if pre and rnd:
            diff = pre[0]["id_acc"] - np.mean([r["id_acc"] for r in rnd], 0)
            m, lo, hi, _, k = boot(diff, rng)
            ratio = pre[0]["id_acc"].mean() / max(np.mean([r["id_acc"].mean() for r in rnd]), 1e-9)
            stage_gain = pre[0]["stage_ba"] - np.mean([r["stage_ba"] for r in rnd])
            finding = lo > 0 and ratio >= 2 and stage_gain <= 0.02
            out += [f"\nSubject-ID accuracy, pretrained minus random-init (as-is, {k} subjects): {m:+.4f} [{lo:+.4f}, {hi:+.4f}]; ratio {ratio:.2f}; "
                    f"stage balanced accuracy gain {stage_gain:+.4f}. **Fingerprint finding: {'yes' if finding else 'no'}.**"]
    pre_a, pre_p = g("pretrained", "asis"), g("pretrained", "perwindow")
    if pre_a and pre_p:
        a_, p_ = pre_a[0]["amp_acc"], pre_p[0]["amp_acc"]
        out += [f"\nAmplitude probe, pretrained: as-is {a_:.3f}, per-window {p_:.3f}. **Amplitude finding: {'yes' if a_ > 0.4 and (p_ - 0.2) <= 0.5 * (a_ - 0.2) else 'no'}.**"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("anchors", "embed", "probe", "finetune", "probe-b", "report"))
    ap.add_argument("--shards", default="/shared/aayu/fill-a/shards/shhs1")
    ap.add_argument("--nsrr", default="/shared/aayu/nsrr/shhs/polysomnography")
    ap.add_argument("--ckpt", default="/shared/aayu/fill-a/runs/kahlus-10m/ckpt-00003139.pt")
    ap.add_argument("--work", default="/shared/aayu/pilot-ab")
    ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--init", choices=("pretrained", "random", "bandpower"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--input", choices=("asis", "perwindow"), default="asis")
    ap.add_argument("--set", choices=("a", "b"), default="a")
    ap.add_argument("--shift", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--steps", type=int, default=4000)
    a = ap.parse_args()
    global FT_STEPS
    FT_STEPS = a.steps
    if a.cmd == "anchors":
        a.out = a.out or a.work
    {"anchors": cmd_anchors, "embed": cmd_embed, "probe": cmd_probe, "finetune": cmd_finetune, "probe-b": cmd_probe_b, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
