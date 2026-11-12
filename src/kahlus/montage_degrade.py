"""Montage degradation for the low-cost headband stress test (issue #97).

Each function turns clean PSG EEG (microvolts) into something closer to a cheap
dry-electrode headband, so an arousal detector's event-level loss can be measured
as a function of signal quality. Seeds make every degradation reproducible.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly


def add_white_noise(x_uv: np.ndarray, rms_uv: float, seed: int = 0) -> np.ndarray:
    """Broadband amplifier and electrode noise at a given RMS (ADS1299 input noise is
    about 1 uV RMS; dry electrodes in practice add several uV)."""
    return x_uv + rms_uv * np.random.default_rng(seed).standard_normal(x_uv.size)


def add_drift_and_pops(x_uv: np.ndarray, fs: float, *, drift_uv: float, pops_per_min: float, seed: int = 0) -> np.ndarray:
    """Slow baseline wander (random walk, low-passed) plus electrode pops: step-like
    transients with exponential recovery, the signature of dry-contact movement."""
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.standard_normal(x_uv.size))
    walk -= np.convolve(walk, np.ones(int(fs * 10)) / int(fs * 10), mode="same")  # keep < ~0.1 Hz wander bounded
    walk *= drift_uv / (walk.std() + 1e-12)
    out = x_uv + walk
    n_pops = rng.poisson(pops_per_min * x_uv.size / fs / 60.0)
    tau = int(0.3 * fs)
    kernel = np.exp(-np.arange(5 * tau) / tau)
    for start in rng.integers(0, max(x_uv.size - kernel.size, 1), n_pops):
        out[start : start + kernel.size] += rng.choice([-1, 1]) * rng.uniform(100, 400) * kernel
    return out


def downsample(x_uv: np.ndarray, fs: float, target_fs: float) -> tuple[np.ndarray, float]:
    """Polyphase resample to target_fs (anti-aliased). Returns the new signal and rate."""
    from fractions import Fraction

    f = Fraction(target_fs / fs).limit_denominator(1000)
    return resample_poly(x_uv, f.numerator, f.denominator), fs * f.numerator / f.denominator


LEVELS = {
    "clean": {},
    "noise_5uv": {"rms_uv": 5.0},
    "noise_10uv": {"rms_uv": 10.0},
    "noise_20uv": {"rms_uv": 20.0},
    "dry_headband": {"rms_uv": 10.0, "drift_uv": 30.0, "pops_per_min": 1.0},
    "dry_headband_bad": {"rms_uv": 20.0, "drift_uv": 60.0, "pops_per_min": 4.0},
}


def degrade(x_uv: np.ndarray, fs: float, level: str, seed: int = 0) -> np.ndarray:
    p = LEVELS[level]
    out = x_uv
    if "drift_uv" in p:
        out = add_drift_and_pops(out, fs, drift_uv=p["drift_uv"], pops_per_min=p["pops_per_min"], seed=seed)
    if "rms_uv" in p:
        out = add_white_noise(out, p["rms_uv"], seed=seed + 1)
    return out


def detected_onsets(prob: np.ndarray, times: np.ndarray, threshold: float, min_gap_s: float = 10.0) -> np.ndarray:
    """Rising edges of prob >= threshold, merged when closer than min_gap_s."""
    above = prob >= threshold
    rises = times[np.flatnonzero(above & ~np.concatenate([[False], above[:-1]]))]
    kept: list[float] = []
    for t in rises:
        if not kept or t - kept[-1] >= min_gap_s:
            kept.append(float(t))
    return np.asarray(kept)


def _match_scan(detected: np.ndarray, truth: np.ndarray, tol_s: float) -> list[float]:
    """Each detection in order takes the nearest unused true onset (lowest index on a
    tie) if it lies within tol_s. Scans every onset per detection."""
    used = np.zeros(truth.size, dtype=bool)
    errors = []
    for d in detected:
        if truth.size == 0:
            break
        gap = np.abs(truth - d)
        gap[used] = np.inf
        j = int(np.argmin(gap))
        if gap[j] <= tol_s:
            used[j] = True
            errors.append(float(d - truth[j]))
    return errors


def _match_window(detected: np.ndarray, truth: np.ndarray, tol_s: float) -> list[float]:
    """_match_scan's matches, looking only at onsets in a sorted window around each
    detection. An onset farther than tol_s never matches, so the nearest unused onset
    matches exactly when the nearest unused onset in the window does. The window is
    1 s wider than tol_s on each side, far more than float rounding can move a gap."""
    order = np.argsort(truth, kind="stable")
    ts = truth[order]
    lo = np.searchsorted(ts, detected - (tol_s + 1.0), side="left")
    hi = np.searchsorted(ts, detected + (tol_s + 1.0), side="right")
    used = np.zeros(truth.size, dtype=bool)
    errors = []
    for d, a, b in zip(detected, lo, hi):
        cand = order[a:b]
        cand = cand[~used[cand]]
        if cand.size == 0:
            continue
        gap = np.abs(truth[cand] - d)
        best = gap.min()
        if best <= tol_s:
            j = int(cand[gap == best].min())
            used[j] = True
            errors.append(float(d - truth[j]))
    return errors


def match_events(detected: np.ndarray, truth: np.ndarray, tol_s: float) -> dict:
    """Greedy one-to-one matching of detected to true onsets within tol_s.

    Each detection, in order, takes the nearest unused true onset (lowest index on a
    tie) if it lies within tol_s. Scanning every onset per detection grows with the
    square of the cohort size, so finite inputs use the sorted-window search, which
    gives the same matches.
    """
    finite = np.isfinite(truth).all() and np.isfinite(detected).all() and np.isfinite(tol_s)
    errors = (_match_window if finite and truth.size else _match_scan)(detected, truth, tol_s)
    tp = len(errors)
    precision = tp / detected.size if detected.size else 0.0
    recall = tp / truth.size if truth.size else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "n_detected": int(detected.size), "n_true": int(truth.size), "precision": precision,
            "recall": recall, "f1": f1, "median_abs_timing_error_s": float(np.median(np.abs(errors))) if errors else float("nan")}
