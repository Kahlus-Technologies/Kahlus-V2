"""Offline closed-loop auditory stimulation (CLAS) replay (issue #96).

A causal slow-oscillation trigger, replayed on recorded EEG. Clicks are never
actually played; the question is where a standard trigger *would* place them
relative to scored arousals, and whether a veto can move them away.

Trigger (Ngo 2013 style, causal): during the last completed epoch scored N2/N3,
filter EEG 0.5-4 Hz with a causal (forward-only) Butterworth, detect a negative
half-wave crossing below ``NEG_UV``, then schedule a click ``UP_DELAY_S`` later
(approximate up-state). One click per ``REFRACTORY_S``.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter

NEG_UV = -40.0
UP_DELAY_S = 0.5
REFRACTORY_S = 2.5


def so_filter(eeg_uv: np.ndarray, fs: float) -> np.ndarray:
    """Causal 0.5-4 Hz band-pass."""
    b, a = butter(2, [0.5 / (fs / 2), 4.0 / (fs / 2)], btype="band")
    return lfilter(b, a, eeg_uv)


def so_clicks(eeg_uv: np.ndarray, fs: float, allowed: np.ndarray, neg_uv: float | np.ndarray = NEG_UV) -> np.ndarray:
    """Click times (s). ``allowed`` is a boolean per-second mask (N2/N3 known at t).
    ``neg_uv`` is the trigger level in microvolts: one number, or one level per second of the night."""
    x = so_filter(eeg_uv, fs)
    level = neg_uv if np.ndim(neg_uv) == 0 else np.asarray(neg_uv)[np.minimum((np.arange(x.size) / fs).astype(int), np.size(neg_uv) - 1)]
    below = x < level
    crossings = np.flatnonzero(below[1:] & ~below[:-1]) + 1  # entering the trough
    clicks, last = [], -np.inf
    for c in crossings:
        t = c / fs + UP_DELAY_S
        sec = int(c / fs)
        if sec < allowed.size and allowed[sec] and t - last >= REFRACTORY_S:
            clicks.append(t)
            last = t
    return np.asarray(clicks)


def clicks_before_arousal(clicks: np.ndarray, onsets: np.ndarray, window_s: float) -> np.ndarray:
    """Boolean per click: a spontaneous arousal starts within (click, click + window_s]."""
    if onsets.size == 0 or clicks.size == 0:
        return np.zeros(clicks.size, dtype=bool)
    j = np.searchsorted(onsets, clicks, side="right")
    nxt = np.where(j < onsets.size, onsets[np.minimum(j, onsets.size - 1)], np.inf)
    return (nxt - clicks) <= window_s


# --- K7: do trigger times come before scored arousals? Matched controls inside the same night (replay only). ---
# Rules follow docs/results/k7_k1_s1_preregistration.md section 1.

CONTROLS_PER_CLICK = 5
RADIUS_S, WIDE_RADIUS_S = 600.0, 1800.0  # controls come from this near the click; widened once when none exist
GUARD_S = 3.0  # a control has no click this close, and no other control of its click this close
SINCE_BINS_S = (120.0, 600.0)  # time since the last arousal onset of any kind: <2 min, 2-10 min, >10 min
RUN_GAP_S = 5.0  # clicks closer than this belong to one run


def since_arousal_bin(arousal_onsets: np.ndarray, n_sec: int) -> np.ndarray:
    """Per second t: 0, 1 or 2 by time since the last arousal onset at or before t (2 when there is none)."""
    t = np.arange(n_sec)
    k = np.searchsorted(np.sort(arousal_onsets), t, side="right") - 1
    since = np.where(k >= 0, t - np.sort(arousal_onsets)[np.maximum(k, 0)] if arousal_onsets.size else np.inf, np.inf)
    return np.searchsorted(SINCE_BINS_S, since, side="right").astype(np.int8)


def _near(times: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Distance from each time to the nearest ``ref`` (inf when ref is empty); ref sorted."""
    if ref.size == 0:
        return np.full(times.shape, np.inf)
    j = np.searchsorted(ref, times)
    return np.minimum(np.abs(times - ref[np.minimum(j, ref.size - 1)]), np.abs(times - ref[np.maximum(j - 1, 0)]))


def matched_controls(
    clicks: np.ndarray, stage_sec: np.ndarray, bin_sec: np.ndarray, *, rng: np.random.Generator,
    avoid: np.ndarray | None = None, match_bin: bool = True, n_per: int = CONTROLS_PER_CLICK,
) -> tuple[np.ndarray, int]:
    """Control moments per click, shape (n_clicks, n_per) with NaN where fewer exist, and the number of clicks dropped.

    A control is a second-plus-random-fraction in the click's night with the click's stage code (0 = trigger
    off, so never), the click's time-since-arousal bin (unless ``match_bin`` is off), within RADIUS_S of the
    click (WIDE_RADIUS_S when none), at least GUARD_S from any ``avoid`` time (default: the clicks) and from
    the click's other controls. A click with no control, even after widening, is dropped (all NaN).
    """
    avoid = np.sort(clicks if avoid is None else avoid)
    out = np.full((clicks.size, n_per), np.nan)
    sec = np.flatnonzero(stage_sec > 0)
    sec = sec[_near(sec + 0.5, avoid) > GUARD_S + 0.5] if avoid.size else sec
    sec_stage, sec_bin = stage_sec[sec], bin_sec[sec]
    dropped = 0
    for i, t in enumerate(clicks):
        sk = min(int(t), stage_sec.size - 1)
        ok = sec_stage == stage_sec[sk]
        if match_bin:
            ok &= sec_bin == bin_sec[sk]
        pool = sec[ok]
        for radius in (RADIUS_S, WIDE_RADIUS_S):
            lo, hi = np.searchsorted(pool, [t - radius, t + radius])
            cand = pool[lo:hi]
            if cand.size:
                break
        else:
            dropped += 1
            continue
        taken: list[float] = []
        for c in rng.permutation(cand) + rng.random(cand.size):
            if all(abs(c - x) >= GUARD_S for x in taken):
                taken.append(c)
                if len(taken) == n_per:
                    break
        out[i, : len(taken)] = taken
    return out, dropped


def first_of_runs(clicks: np.ndarray) -> np.ndarray:
    """Keep the first click of each run of clicks less than RUN_GAP_S apart."""
    return clicks[np.concatenate([[True], np.diff(clicks) >= RUN_GAP_S])] if clicks.size else clicks


def rotate_clicks(clicks: np.ndarray, stage_sec: np.ndarray) -> np.ndarray:
    """Shift control: move click times by half the night (wrapping); keep those where the trigger is allowed
    and no real click lies within GUARD_S."""
    t = np.sort((clicks + stage_sec.size / 2) % stage_sec.size)
    keep = (stage_sec[np.minimum(t.astype(int), stage_sec.size - 1)] > 0) & (_near(t, clicks) > GUARD_S)
    return t[keep]


def running_level(x: np.ndarray, fs: float, pct: float = 90.0, default_uv: float = NEG_UV) -> np.ndarray:
    """Per second: minus the ``pct`` percentile of |x| over the samples before that second (1 Hz thinned), a
    trigger level that follows the night's own amplitude. ``default_uv`` applies before any sample."""
    import bisect

    step = max(int(fs), 1)
    a = np.abs(x[::step])
    seen: list[float] = []
    out = np.full(a.size, default_uv)
    for s in range(a.size):
        if seen:
            out[s] = -seen[min(int(len(seen) * pct / 100.0), len(seen) - 1)]
        bisect.insort(seen, float(a[s]))
    return out


def hr_change(beat_times: np.ndarray, rr: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Mean heart rate (bpm) in (t + 2, t + 10] minus the mean in (t - 10, t]; NaN with under 3 beats in a window.
    ECG only, needs no scorer. The later window reads past t on purpose: this is an outcome, never a feature."""
    c = np.concatenate([[0.0], np.cumsum(60.0 / rr)])
    def mean(a: float, b: float) -> np.ndarray:
        i0, i1 = np.searchsorted(beat_times, t + a, side="right"), np.searchsorted(beat_times, t + b, side="right")
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(i1 - i0 >= 3, (c[i1] - c[i0]) / (i1 - i0), np.nan)
    return mean(2.0, 10.0) - mean(-10.0, 0.0)


def subject_bootstrap(per_subject: np.ndarray, stat, *, n_boot: int = 2000, seed: int = 0) -> dict:
    """Estimate, 95% percentile interval and two-sided bootstrap p-value (null: ``null``) of ``stat(column sums)``
    resampling whole subjects; ``per_subject`` is (subjects, k). ``stat`` is ``(fn, null)``."""
    fn, null = stat
    pick = np.random.default_rng(seed).integers(0, len(per_subject), size=(n_boot, len(per_subject)))
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = fn(per_subject[pick].sum(1).T)
    boot = boot[np.isfinite(boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5]) if boot.size else (np.nan, np.nan)
    p = 2 * min((boot <= null).mean(), (boot >= null).mean()) if boot.size else np.nan
    return {"estimate": float(fn(per_subject.sum(0))), "ci_low": float(lo), "ci_high": float(hi), "p": float(max(p, 1 / (boot.size + 1)))}


#: Statistics over per-subject sums (events at clicks, events at controls, matched clicks).
RELATIVE_RISK = (lambda c: c[0] / c[1], 1.0)
MEAN_DIFFERENCE = (lambda c: (c[0] - c[1]) / c[2], 0.0)
