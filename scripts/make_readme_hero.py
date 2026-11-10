"""Render the README hero image from a real NSRR night.

Official Kahlus wordmark on a dark field, over three real traces from one
MESA recording around a scored spontaneous arousal: C4 EEG, ECG, and the beat
interval series from the causal QRS detector. Nothing is synthetic; the night,
channel and onset are printed in the corner so the figure can be checked.

Usage: python scripts/make_readme_hero.py ~/data/nsrr/mesa --out docs/assets/hero.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.signal import butter, sosfiltfilt  # noqa: E402

from kahlus.nsrr import classify_arousals, discover_nsrr_recordings, load_nsrr_epoch_matrix, parse_nsrr_events  # noqa: E402
from kahlus.qrs_fast import causal_beats  # noqa: E402

BG, INK, DIM, ACCENT = "#0B0B0C", "#F5F5F5", "#5C5C60", "#E8E8E8"
PRE_S, POST_S = 20.0, 10.0


def band(x: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    return sosfiltfilt(butter(4, [lo, hi], btype="band", fs=fs, output="sos"), x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesa", type=Path)
    ap.add_argument("--out", type=Path, default=Path("docs/assets/hero.png"))
    ap.add_argument("--night", default="0002")
    ap.add_argument("--arousal-index", type=int, default=35, help="which clean spontaneous arousal of the night")
    a = ap.parse_args()

    rec = next(r for r in discover_nsrr_recordings(a.mesa, dataset="mesa") if r.subject_id == a.night)
    m = load_nsrr_epoch_matrix(rec.edf_path, channel_groups={"eeg": ("EEG3",), "ecg": ("EKG",)})
    fs = m["sfreq"]
    eeg = m["epochs"]["eeg"][:, :, 0].ravel() * 1e6
    ecg = m["epochs"]["ecg"][:, :, 0].ravel()
    onset = [x.onset for x in classify_arousals(parse_nsrr_events(rec.xml_path)) if x.clean][a.arousal_index]
    lo, hi = int((onset - PRE_S) * fs), int((onset + POST_S) * fs)
    t = np.arange(lo, hi) / fs - onset
    e = band(eeg[lo - int(5 * fs) : hi + int(5 * fs)], fs, 0.5, 35)[int(5 * fs) : -int(5 * fs)]
    k = band(ecg[lo - int(5 * fs) : hi + int(5 * fs)], fs, 5, 30)[int(5 * fs) : -int(5 * fs)]
    bt, rr = causal_beats(ecg, fs)
    sel = (bt >= onset - PRE_S) & (bt <= onset + POST_S)

    fig = plt.figure(figsize=(16, 6), dpi=200, facecolor=BG)
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.58], facecolor=BG)
    norm = lambda x: (x - np.median(x)) / (np.percentile(np.abs(x - np.median(x)), 99) + 1e-9)  # noqa: E731
    ax.plot(t, norm(e) * 0.9 + 2.2, color=INK, lw=0.7, alpha=0.95)
    ax.plot(t, norm(k) * 0.5 + 0.9, color=INK, lw=0.6, alpha=0.6)
    hr = 60.0 / rr[sel]
    ax.plot(bt[sel] - onset, (hr - hr.min()) / (np.ptp(hr) + 1e-9) * 0.6 - 0.55, color=ACCENT, lw=1.6, marker="o", ms=2.5)
    ax.text(POST_S - 0.3, 0.12, f"{hr.min():.0f} to {hr.max():.0f} bpm", color=DIM, fontsize=9, family="monospace", ha="right")
    ax.axvline(0, color=INK, lw=0.8, ls=(0, (2, 3)), alpha=0.7)
    for y, label in [(2.95, "EEG  C4"), (1.45, "ECG"), (0.25, "beat-to-beat HR")]:
        ax.text(-PRE_S + 0.3, y, label, color=DIM, fontsize=9, family="monospace")
    ax.text(0.25, 2.95, "scored arousal onset", color=DIM, fontsize=9, family="monospace")
    ax.text(POST_S - 0.3, -0.95, f"MESA {a.night}  ·  real recording  ·  t = {onset:.0f} s", color=DIM, fontsize=8,
            family="monospace", ha="right")
    ax.set_xlim(-PRE_S, POST_S)
    ax.set_ylim(-1.05, 3.15)
    ax.axis("off")

    mark = Image.open(Path(__file__).resolve().parents[1] / "docs/assets/kahlus-wordmark-light.png")
    # High-res wordmark (docs/assets/kahlus-wordmark-light.png), drawn at a fixed share of the width.
    target_w = int(fig.bbox.width * 0.42)
    big = mark.resize((target_w, round(mark.height * target_w / mark.width)), Image.LANCZOS)
    fig.figimage(np.asarray(big), xo=int((fig.bbox.width - big.width) / 2), yo=int(fig.bbox.height * 0.735), origin="upper")
    fig.text(0.5, 0.655, "SENSE  ·  UNDERSTAND  ·  RECOVER", color=DIM, ha="center", fontsize=11, family="monospace")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, facecolor=BG)
    print(a.out, f"night {a.night}, onset {onset:.1f} s, beats in window {int(sel.sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
