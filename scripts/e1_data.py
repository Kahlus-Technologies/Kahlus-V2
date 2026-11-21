"""E1 data lists (docs/prereg/e1.md, section 2.2): training pools r1, r4, r16 and the quality cells clean4 and dirty4.

    python scripts/e1_data.py --shards /shared/aayu/fill-a/shards/shhs1-full --nsrr /shared/aayu/nsrr/shhs/polysomnography --out /shared/aayu/e1

Per night, from the EDF (C4 and ECG), a 1 s patch is bad if it is flat (std under 1% of the night's median patch std of that channel),
railed (a sample within 0.5% of the header's physical minimum or maximum) or a burst (C4 peak-to-peak over 10 times the night's median).
q = 1 - bad fraction. P = shard nights that are neither held out nor dev nights. Per quartile of the arousal index (arousals per hour of
sleep), clean4 is the top 25% by q and dirty4 the bottom 25%, ties by u, so both have the same arousal-index mix. Per-night files stay
under --out; stdout carries only counts, hours and the gate GQ numbers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from kahlus.nsrr import SLEEP_STAGES, event_kind, load_nsrr_epoch_matrix, parse_nsrr_events
from kahlus.shards import FS_HZ, ShardSet, is_heldout

DEV = range(200001, 200026)
UNIT = {"uV": 1e-6, "mV": 1e-3, "V": 1.0}


def u_of(subject: str) -> float:
    return int(hashlib.sha256(f"e1:shhs:{subject}".encode()).hexdigest(), 16) / 2**256


def edf_ranges(path: Path) -> dict[str, tuple[float, float]]:
    """Physical (min, max) in volts per signal label, straight from the EDF header."""
    with open(path, "rb") as f:
        head = f.read(256)
        ns = int(head[252:256])
        body = f.read(ns * 256)
    field = lambda off, width: [body[off + i * width: off + (i + 1) * width].decode("latin1").strip() for i in range(ns)]  # noqa: E731
    labels, dims = field(0, 16), field(ns * 96, 8)
    lo, hi = field(ns * 104, 8), field(ns * 112, 8)
    return {lab: (float(a) * UNIT.get(d, 1.0), float(b) * UNIT.get(d, 1.0)) for lab, d, a, b in zip(labels, dims, lo, hi)}


def one_night(job):
    subject, edf, xml = job
    m = load_nsrr_epoch_matrix(edf, channel_groups={"eeg": ("EEG",), "ecg": ("ECG",)})
    rng = edf_ranges(edf)
    bad = None
    for g, label in (("eeg", "EEG"), ("ecg", "ECG")):
        x = m["epochs"][g][:, :, 0].ravel()
        fs = int(round(m["sfreq"]))
        p = x[: x.size // fs * fs].reshape(-1, fs)
        sd, ptp = p.std(1), p.max(1) - p.min(1)
        lo, hi = rng.get(label, (p.min(), p.max()))
        margin = 0.005 * (hi - lo)
        b = (sd < 0.01 * np.median(sd)) if np.median(sd) > 0 else np.zeros(len(p), bool)
        b |= (p >= hi - margin).any(1) | (p <= lo + margin).any(1)
        if g == "eeg":
            b |= ptp > 10 * np.median(ptp)
        bad = b if bad is None else bad[: len(b)] | b[: len(bad)]
    ev = parse_nsrr_events(xml)
    sleep_s = sum(d for c, _, d in ev if c in SLEEP_STAGES)
    arousals = sum(event_kind(c) == "arousal" for c, _, _ in ev)
    return subject, float(bad.mean()), arousals / max(sleep_s / 3600.0, 1e-3), sleep_s / 3600.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shards", default="/shared/aayu/fill-a/shards/shhs1-full")
    ap.add_argument("--nsrr", default="/shared/aayu/nsrr/shhs/polysomnography")
    ap.add_argument("--out", default="/shared/aayu/e1")
    ap.add_argument("--workers", type=int, default=60)
    a = ap.parse_args(argv)
    shards = ShardSet(a.shards)
    pool = [n for n in shards.nights if n["dataset"] == "shhs" and not is_heldout("shhs", n["subject"]) and int(re.sub(r"\D", "", n["subject"])) not in DEV]
    root = Path(a.nsrr)
    jobs = [(n["subject"], root / f"edfs/shhs1/shhs1-{n['subject']}.edf", root / f"annotations-events-nsrr/shhs1/shhs1-{n['subject']}-nsrr.xml") for n in pool]
    with Pool(a.workers) as p:
        rows = p.map(one_night, jobs, chunksize=4)
    out = Path(a.out)
    (out / "lists").mkdir(parents=True, exist_ok=True)
    hours = {n["subject"]: n["n"] / 3600 for n in pool}
    subj = np.asarray([r[0] for r in rows])
    bad = np.asarray([r[1] for r in rows])
    ai = np.asarray([r[2] for r in rows])
    u = np.asarray([u_of(s) for s in subj])
    (out / "night_stats.csv").write_text("subject,bad_fraction,arousal_index,sleep_h,u\n" + "".join(
        f"{s},{b:.6f},{x:.4f},{r[3]:.3f},{uu:.6f}\n" for s, b, x, r, uu in zip(subj, bad, ai, rows, u)))
    cells = {"r1": np.ones(len(subj), bool), "r4": u < 0.25, "r16": u < 1 / 16}
    q = 1 - bad
    quart = np.digitize(ai, np.quantile(ai, [0.25, 0.5, 0.75]))
    clean, dirty = np.zeros(len(subj), bool), np.zeros(len(subj), bool)
    for k in range(4):
        idx = np.flatnonzero(quart == k)
        k4 = max(1, len(idx) // 4)
        clean[idx[np.lexsort((u[idx], -q[idx]))[:k4]]] = True
        dirty[idx[np.lexsort((u[idx], q[idx]))[:k4]]] = True
    cells |= {"clean4": clean, "dirty4": dirty}
    for name, m in cells.items():
        (out / "lists" / f"{name}.txt").write_text("\n".join(subj[m]) + "\n")
        print(f"{name}: {int(m.sum())} nights, {sum(hours[s] for s in subj[m]):.0f} h, median bad fraction {np.median(bad[m]):.4f}")
    med_c, med_d = float(np.median(bad[clean])), float(np.median(bad[dirty]))
    gq = med_d >= 3 * med_c and med_d - med_c >= 0.01
    print(f"GQ {'PASS' if gq else 'FAIL'}: median bad fraction dirty4 {med_d:.4f}, clean4 {med_c:.4f}, ratio {med_d / max(med_c, 1e-9):.1f}, "
          f"difference {med_d - med_c:.4f}")
    (out / "gq.json").write_text(json.dumps({"pass": bool(gq), "dirty4": med_d, "clean4": med_c}))
    print(f"arousal index quartile edges {np.quantile(ai, [0.25, 0.5, 0.75]).round(1).tolist()}; sampling rate {FS_HZ} Hz")


if __name__ == "__main__":
    main()
