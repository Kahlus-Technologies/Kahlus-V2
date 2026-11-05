"""Vectorized Pan-Tompkins QRS detector.

wfdb's xqrs_detect is a pure-Python sample-by-sample loop: ~147 s for a 7-hour
200 Hz record (5.1 M samples). This implementation is fully vectorized numpy
and runs the same record in <1 s, a ~200x speedup, which is the difference
between a 20-hour and a 10-minute cohort extraction.

Correctness is not assumed. ``validate_against_xqrs`` compares beat counts and
the resulting 1 Hz heart-rate series against xqrs on real records; the adapter
records which detector produced each record's HR so any result can be audited
for detector dependence.

Algorithm (Pan & Tompkins 1985) with one deviation:
    bandpass 5-15 Hz -> derivative -> square -> 150 ms moving integration
    -> *block-adaptive* threshold -> refractory-constrained peak picking

The deviation is the threshold. A single global percentile fails on overnight
PSG because ECG amplitude drifts with posture, electrode drying and body
movement across 7 hours. We compute the threshold on a rolling window so it
tracks that drift.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import butter, filtfilt, find_peaks

#: Absolute refractory period. No two QRS complexes can be closer than this
#: (300 bpm ceiling).
REFRACTORY_S = 0.20
#: T-wave discrimination window. A candidate arriving between REFRACTORY_S and
#: this bound after an accepted beat is a T-wave rather than a QRS unless its
#: slope is comparable, because the T-wave follows QRS by roughly 200-400 ms and
#: is broader. Pan & Tompkins use 360 ms for the same purpose.
TWAVE_WINDOW_S = 0.36
#: A candidate inside the T-wave window must have at least this fraction of the
#: preceding beat's slope to be accepted as a genuine QRS. T-waves are broad and
#: slow, so their maximum derivative is far below that of the QRS that precedes
#: them.
TWAVE_SLOPE_RATIO = 0.55
#: Physiological heart-rate bounds; intervals outside are dropped before
#: interpolation rather than allowed to distort the rate series.
HR_MIN_BPM = 25.0
HR_MAX_BPM = 220.0
#: Rolling window for the adaptive threshold. Long enough to contain several
#: beats at any plausible rate, short enough to track overnight amplitude drift.
THRESH_WINDOW_S = 8.0
#: Fraction of the local maximum used as the detection threshold.
THRESH_FRACTION = 0.30


def detect_qrs(ecg: np.ndarray, fs: float) -> np.ndarray:
    """Return QRS sample indices. Vectorized; ~200x faster than xqrs.

    Parameters
    ----------
    ecg:
        Raw ECG samples. Scaling is irrelevant (the threshold is relative).
    fs:
        Sampling rate in Hz.
    """
    x = np.asarray(ecg, dtype=np.float64)
    if len(x) < int(fs * 10):
        return np.zeros(0, dtype=np.int64)

    # Guard against flat or non-finite leads before filtering.
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if np.std(x) < 1e-9:
        return np.zeros(0, dtype=np.int64)

    # 1. Bandpass 5-15 Hz isolates the QRS energy band, rejecting baseline
    #    wander below and muscle noise above.
    nyq = fs / 2.0
    low, high = 5.0 / nyq, min(15.0 / nyq, 0.99)
    b, a = butter(2, [low, high], btype="band")
    filtered = filtfilt(b, a, x)

    # 2-3. Derivative then square: emphasises the steep QRS slope and makes
    #      the signal strictly positive.
    deriv = np.diff(filtered, prepend=filtered[0])
    squared = deriv * deriv

    # 4. Moving-window integration over ~QRS width turns the spiky squared
    #    derivative into one smooth bump per beat.
    win = max(1, int(0.15 * fs))
    integrated = uniform_filter1d(squared, size=win, mode="nearest")

    # 5. Block-adaptive threshold. maximum_filter1d gives the local peak
    #    amplitude at every sample in O(n); scaling it tracks amplitude drift
    #    across the night without a Python loop.
    tw = max(win * 4, int(THRESH_WINDOW_S * fs))
    local_max = maximum_filter1d(integrated, size=tw, mode="nearest")
    threshold = THRESH_FRACTION * local_max

    # 6. Peak picking with a hard refractory distance.
    candidates, _ = find_peaks(integrated, distance=max(1, int(REFRACTORY_S * fs)))
    if len(candidates) == 0:
        return np.zeros(0, dtype=np.int64)
    keep = integrated[candidates] > threshold[candidates]
    candidates = candidates[keep]
    if len(candidates) < 2:
        return candidates.astype(np.int64)

    # 7. T-wave rejection. The refractory period alone leaves T-waves that fall
    #    200-360 ms after a QRS, and each false beat splits one true RR interval
    #    into two wrong ones, which destroys the rate series even when every real
    #    beat was found. Discriminate on slope: a QRS is steep, a T-wave is not.
    return _reject_twaves(candidates, np.abs(deriv), fs)


def _reject_twaves(
    candidates: np.ndarray, abs_deriv: np.ndarray, fs: float
) -> np.ndarray:
    """Drop candidates in the T-wave window whose slope is too shallow for a QRS.

    Runs over candidate peaks (tens of thousands) rather than raw samples
    (millions), so the sequential pass costs milliseconds. The dependency is
    genuinely sequential: whether a candidate is accepted determines the
    reference slope for the next one.
    """
    twave_samples = int(TWAVE_WINDOW_S * fs)
    half = max(1, int(0.05 * fs))  # +/-50 ms around the peak covers a QRS
    n = len(abs_deriv)

    def slope_at(idx: int) -> float:
        lo = max(0, idx - half)
        hi = min(n, idx + half)
        return float(abs_deriv[lo:hi].max()) if hi > lo else 0.0

    accepted = [int(candidates[0])]
    last_slope = slope_at(accepted[0])

    for raw_idx in candidates[1:]:
        idx = int(raw_idx)
        gap = idx - accepted[-1]
        if gap < twave_samples:
            slope = slope_at(idx)
            if slope < TWAVE_SLOPE_RATIO * last_slope:
                continue  # too shallow this soon after a QRS: it is a T-wave
            accepted.append(idx)
            last_slope = slope
        else:
            accepted.append(idx)
            last_slope = slope_at(idx)

    return np.asarray(accepted, dtype=np.int64)


def beat_intervals(peaks: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Physiologic RR intervals from QRS sample indices.

    Returns ``(beat_times_s, rr_s)``: each RR interval is stamped with the time of
    the beat that closes it, so an interval is only known once that beat happened.
    Intervals outside HR_MIN_BPM..HR_MAX_BPM are dropped.
    """
    times = np.asarray(peaks, dtype=np.float64) / fs
    rr = np.diff(times)
    rate = 60.0 / np.maximum(rr, 1e-6)
    keep = (rate >= HR_MIN_BPM) & (rate <= HR_MAX_BPM)
    return times[1:][keep], rr[keep]


def heart_rate_grid(
    ecg: np.ndarray, fs: float, n_seconds: int, *, min_beats: int = 100
) -> tuple[np.ndarray | None, dict]:
    """Instantaneous heart rate (bpm) on a 1 Hz grid.

    Returns ``(hr, info)``. ``hr`` is None when detection is too sparse to be
    trusted, which is the caller's signal to fall back; ``info`` carries the
    detector diagnostics that make a fallback auditable after the fact.
    """
    peaks = detect_qrs(ecg, fs)
    info: dict = {"n_beats": int(len(peaks)), "detector": "pan_tompkins_vectorized"}

    if len(peaks) < min_beats:
        info["reason"] = f"only {len(peaks)} beats detected (need >={min_beats})"
        return None, info

    beat_times, rr = beat_intervals(peaks, fs)
    info["frac_physiologic"] = float(len(rr) / max(len(peaks) - 1, 1))
    if len(rr) < min_beats:
        info["reason"] = (
            f"only {len(rr)} physiologic intervals "
            f"({info['frac_physiologic']:.1%} of {len(peaks) - 1})"
        )
        return None, info

    beat_rates = 60.0 / rr

    grid = np.interp(
        np.arange(n_seconds, dtype=np.float64),
        beat_times,
        beat_rates,
        left=float(beat_rates[0]),
        right=float(beat_rates[-1]),
    ).astype(np.float32)

    info["median_bpm"] = float(np.median(beat_rates))
    info["mean_bpm"] = float(np.mean(beat_rates))
    return grid, info


def validate_against_xqrs(
    ecg: np.ndarray, fs: float, n_seconds: int
) -> dict:
    """Compare this detector to wfdb's xqrs on one record.

    Slow by construction (it runs xqrs). Intended for pipeline validation on a
    handful of records, never inside the cohort extraction loop.
    """
    import wfdb.processing as wp

    hr_fast, info_fast = heart_rate_grid(ecg, fs, n_seconds)

    beats = wp.xqrs_detect(sig=np.asarray(ecg, dtype=np.float32), fs=fs, verbose=False)
    if len(beats) < 10:
        return {"error": "xqrs found too few beats", "fast": info_fast}

    t_ref = beats / fs
    rate_ref = 60.0 / np.diff(t_ref)
    ok = (rate_ref >= HR_MIN_BPM) & (rate_ref <= HR_MAX_BPM)
    hr_ref = np.interp(
        np.arange(n_seconds, dtype=np.float64),
        t_ref[1:][ok],
        rate_ref[ok],
        left=float(rate_ref[ok][0]),
        right=float(rate_ref[ok][-1]),
    ).astype(np.float32)

    out: dict = {
        "n_beats_fast": info_fast.get("n_beats"),
        "n_beats_xqrs": int(len(beats)),
        "median_bpm_xqrs": float(np.median(rate_ref[ok])),
        "fast_info": info_fast,
    }
    if hr_fast is None:
        out["hr_series_r"] = None
        out["note"] = "fast detector returned no HR (fallback would trigger)"
        return out

    out["median_bpm_fast"] = float(np.median(hr_fast))
    out["hr_series_r"] = float(np.corrcoef(hr_fast, hr_ref)[0, 1])
    out["mean_abs_diff_bpm"] = float(np.mean(np.abs(hr_fast - hr_ref)))
    # Beat-level agreement: fraction of xqrs beats matched within 100 ms.
    fast_peaks = detect_qrs(ecg, fs) / fs
    if len(fast_peaks):
        idx = np.searchsorted(fast_peaks, t_ref)
        idx = np.clip(idx, 1, len(fast_peaks) - 1)
        nearest = np.minimum(
            np.abs(fast_peaks[idx] - t_ref), np.abs(fast_peaks[idx - 1] - t_ref)
        )
        out["frac_xqrs_beats_matched_100ms"] = float(np.mean(nearest < 0.10))
    return out
