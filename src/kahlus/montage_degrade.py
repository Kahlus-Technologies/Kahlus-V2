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
