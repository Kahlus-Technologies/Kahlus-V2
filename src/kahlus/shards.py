"""Memory-mapped patch shards for pretraining (issue #11).

One build pass turns each night into float16 patches shaped ``(n_patches, n_channels, patch_len)``
and packs the nights into large ``.npy`` shards. Training reads random windows from the shards
through ``np.load(mmap_mode="r")``, so no EDF is parsed while a GPU waits.

Everything stored is causal: resampling reads only past samples, and each patch is z-scored against
the earlier patches of the same night. A window cut from a shard therefore never contains
information from after its own last sample.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from kahlus.nsrr import gate_mesa_ids

SCHEMA = 1
FS_HZ = 125  # configs/data/nsrr_common_montage.yaml: target_fs_hz
PATCH_LEN = 125  # 1 s patches
CHANNELS = ("eeg_c4", "ecg")  # the two derivations MESA and SHHS share (heart_arousal.COHORTS)
WARMUP_PATCHES = 60  # patches dropped while the running statistics settle
CLIP = 20.0  # z-scores beyond this are artifacts; the clip also keeps float16 exact enough
HELDOUT_MOD = 10  # one subject in ten is held out of every fit


def causal_resample(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Resample without look-ahead. The low-pass is a one-sided FIR (a delay, never a preview), and the
    linear interpolation runs one input sample late, so output sample k reads only inputs at or before
    its own time."""
    if fs_in == fs_out:
        return x
    from scipy.signal import firwin, lfilter

    if fs_out < fs_in:
        x = lfilter(firwin(101, 0.45 * fs_out, fs=fs_in), 1.0, x)
    t = np.arange(int(x.size * fs_out / fs_in)) * (fs_in / fs_out) - 1.0
    return np.interp(t, np.arange(x.size), x)


def causal_patches(signals: np.ndarray, patch_len: int = PATCH_LEN) -> np.ndarray:
    """``(n_channels, n_samples)`` to float16 ``(n_patches, n_channels, patch_len)``. Patch k is z-scored
    with the mean and SD of patches 0..k-1 of the same channel, then clipped. The first WARMUP_PATCHES
    are dropped."""
    c, n = signals.shape
    p = signals[:, : n // patch_len * patch_len].reshape(c, -1, patch_len).astype(np.float64)
    if p.shape[1] <= WARMUP_PATCHES:
        return np.zeros((0, c, patch_len), dtype=np.float16)
    count = np.arange(1, p.shape[1] + 1) * patch_len
    mu = np.cumsum(p.sum(-1), axis=1) / count
    sd = np.sqrt(np.maximum(np.cumsum((p * p).sum(-1), axis=1) / count - mu * mu, 0.0))
    scale = np.where(sd > 0, sd, np.inf)  # a flat channel becomes zeros, not NaN
    z = (p[:, 1:] - mu[:, :-1, None]) / scale[:, :-1, None]
    return np.clip(z[:, WARMUP_PATCHES - 1 :], -CLIP, CLIP).transpose(1, 0, 2).astype(np.float16)


def is_heldout(dataset: str, subject: str) -> bool:
    """Subject-held-out rule: fixed by a hash of the ID, so it cannot drift between training and scoring."""
    return int(hashlib.sha256(f"{dataset}:{subject}".encode()).hexdigest(), 16) % HELDOUT_MOD == 0


def write_shards(nights, out_dir: str | Path, *, channels=CHANNELS, shard_patches: int = 1 << 21) -> dict:
    """Pack ``(dataset, subject, patches)`` items into shards of about ``shard_patches`` patches and write
    ``index.json``. A night always stays inside one shard."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index = {"schema": SCHEMA, "fs_hz": FS_HZ, "patch_len": None, "channels": list(channels), "nights": []}
    buf: list[np.ndarray] = []
    held = shard = 0

    def flush() -> None:
        nonlocal buf, held, shard
        if buf:
            np.save(out / f"shard-{shard:05d}.npy", np.concatenate(buf))
            buf, held, shard = [], 0, shard + 1

    for dataset, subject, patches in nights:
        if patches.shape[0] == 0:
            continue
        if patches.dtype != np.float16 or patches.shape[1] != len(channels):
            raise ValueError(f"{dataset}/{subject}: expected float16 (n, {len(channels)}, patch_len), got {patches.dtype} {patches.shape}")
        index["patch_len"] = index["patch_len"] or int(patches.shape[2])
        index["nights"].append({"dataset": dataset, "subject": str(subject), "shard": shard, "offset": held, "n": int(patches.shape[0])})
        buf.append(patches)
        held += patches.shape[0]
        if held >= shard_patches:
            flush()
    flush()
    (out / "index.json").write_text(json.dumps(index))
    return index


class ShardSet:
    """Read side. MESA nights pass through the same seal as every other MESA reader (`gate_mesa_ids`)."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        raw = (self.root / "index.json").read_bytes()
        self.index = json.loads(raw)
        if self.index.get("schema") != SCHEMA:
            raise ValueError(f"{self.root}: shard schema {self.index.get('schema')} != {SCHEMA}; rebuild the shards")
        self.sha256 = hashlib.sha256(raw).hexdigest()
        nights = self.index["nights"]
        mesa = sorted({n["subject"] for n in nights if n["dataset"] == "mesa"})
        allowed = set(gate_mesa_ids(mesa)) if mesa else set()
        self.nights = [n for n in nights if n["dataset"] != "mesa" or n["subject"] in allowed]

    def select(self, datasets, *, heldout: bool | None = None) -> list[dict]:
        """Nights of the given datasets. ``heldout`` True/False keeps only held-out / only fit subjects."""
        keep = set(datasets)
        return [n for n in self.nights
                if n["dataset"] in keep and (heldout is None or is_heldout(n["dataset"], n["subject"]) == heldout)]

    @lru_cache(maxsize=64)  # each worker keeps a few shards mapped; the page cache does the rest
    def _shard(self, i: int) -> np.ndarray:
        return np.load(self.root / f"shard-{i:05d}.npy", mmap_mode="r")

    def read(self, night: dict, start: int, length: int) -> np.ndarray:
        if start < 0 or start + length > night["n"]:
            raise IndexError(f"window [{start}, {start + length}) outside night of {night['n']} patches")
        lo = night["offset"] + start
        return self._shard(night["shard"])[lo : lo + length]


class Windows:
    """Uniform draws over every ``length``-patch window of a set of nights."""

    def __init__(self, shards: ShardSet, nights: list[dict], length: int):
        self.shards, self.length = shards, length
        self.nights = [n for n in nights if n["n"] >= length]
        if not self.nights:
            raise ValueError(f"no night has {length} patches")
        self.cum = np.cumsum([n["n"] - length + 1 for n in self.nights])

    def draw(self, rng: np.random.Generator, size: int) -> list[tuple[dict, int]]:
        flat = np.sort(rng.integers(0, self.cum[-1], size=size))  # sorted: neighbouring reads hit the same pages
        which = np.searchsorted(self.cum, flat, side="right")
        starts = flat - np.concatenate([[0], self.cum])[which]
        return [(self.nights[i], int(s)) for i, s in zip(which, starts)]

    def batch(self, rng: np.random.Generator, size: int) -> np.ndarray:
        return np.stack([self.shards.read(n, s, self.length) for n, s in self.draw(rng, size)])


def synthetic_nights(n_nights: int = 24, n_patches: int = 260, *, datasets=("synth_a", "synth_b"), seed: int = 0):
    """Forecastable fake nights for CPU proofs: per-subject oscillators with drifting phase plus noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_patches * PATCH_LEN) / FS_HZ
    for i in range(n_nights):
        freqs = rng.integers(1, 4, size=len(CHANNELS))  # whole Hz: each 1 s patch resembles the one before it
        drift = np.cumsum(rng.normal(0, 0.02, size=(len(CHANNELS), t.size)), axis=1)
        sig = np.sin(2 * np.pi * freqs[:, None] * t + drift) + rng.normal(0, 0.3, size=drift.shape)
        yield datasets[i % len(datasets)], f"s{i:04d}", causal_patches(sig)


def _nsrr_night(job: tuple[str, str, str]) -> tuple[str, str, np.ndarray | str]:
    from kahlus.heart_arousal import COHORTS
    from kahlus.nsrr import load_nsrr_epoch_matrix

    cohort, subject, edf = job
    labels = COHORTS[cohort]
    try:
        # shortcut: MNE resamples a mixed-rate EEG/ECG pair to the higher rate with its own interpolator
        # before causal_resample runs; make that step causal too if such nights enter a claim.
        m = load_nsrr_epoch_matrix(edf, channel_groups={"eeg": labels["eeg_labels"], "ecg": labels["ecg_labels"]})
        sig = np.stack([causal_resample(m["epochs"][g][:, :, 0].ravel(), m["sfreq"], FS_HZ) for g in ("eeg", "ecg")])
    except (KeyError, ValueError, OSError) as exc:
        return cohort, subject, f"{type(exc).__name__}: {exc}"
    return cohort, subject, causal_patches(sig)


def nsrr_nights(cohorts, *, jobs: int = 1):
    """Patches for every discoverable night of each cohort. Discovery applies the MESA seal."""
    from multiprocessing import Pool

    from kahlus.nsrr import discover_nsrr_recordings, nsrr_root

    todo = [(c, r.subject_id, str(r.edf_path)) for c in cohorts for r in discover_nsrr_recordings(nsrr_root(c), dataset=c)]
    with Pool(jobs) if jobs > 1 else contextlib.nullcontext() as pool:
        for cohort, subject, out in pool.imap(_nsrr_night, todo) if pool else map(_nsrr_night, todo):
            if isinstance(out, str):
                print(f"skipped {cohort}/{subject}: {out}")
            else:
                yield cohort, subject, out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", choices=("nsrr", "synthetic"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--cohorts", nargs="+", default=["mesa", "shhs"], choices=("mesa", "shhs"))
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--nights", type=int, default=24, help="synthetic only")
    args = ap.parse_args(argv)
    nights = nsrr_nights(args.cohorts, jobs=args.jobs) if args.source == "nsrr" else synthetic_nights(args.nights)
    index = write_shards(nights, args.out)
    print(f"{len(index['nights'])} nights, {sum(n['n'] for n in index['nights'])} patches -> {args.out}")


if __name__ == "__main__":
    main()
