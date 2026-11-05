"""Heart-only arousal forecasting benchmark (issue #83).

Task: at anchor time t, using only data up to t, predict whether the next arousal
onset in (t, t + h] is spontaneous. Everything is defined so no input can see past t:

* Anchors sit every ``STEP_S`` seconds where the last *completed* 30 s epoch was
  sleep and no arousal, respiratory event, limb movement or wake epoch started in
  the ``QUIET_S`` before t. The epoch that contains t is never read, because its
  score was assigned by a human who saw the signal after t.
* Label per horizon: 1 if the first arousal onset in (t, t + h] is spontaneous,
  0 if no arousal starts in (t, t + h], -1 (excluded) if the first is respiratory
  or limb-related.
"""
from __future__ import annotations

import numpy as np

from neurotwin.adapters.nsrr import SLEEP_STAGES, classify_arousals, sleep_stage_epochs

HORIZONS_S = (5, 15, 30, 60, 120)
STEP_S = 5.0
QUIET_S = 60.0
EPOCH_S = 30.0


def _disturbance_times(events: list[tuple[str, float, float]]) -> np.ndarray:
    """When each disturbance becomes knowable: event onset, except a scored Wake
    run, which is only known once its first 30 s epoch is over. NSRR merges
    consecutive same-stage epochs into one event, so Wake events can be long."""
    times = []
    for concept, start, _ in events:
        c = concept.lower()
        if concept == "Wake":
            times.append(start + EPOCH_S)
        elif "arousal" in c or "apnea" in c or "hypopnea" in c or "limb" in c or "leg movement" in c:
            times.append(start)
    return np.sort(np.asarray(times, dtype=np.float64))


def anchor_labels(
    events: list[tuple[str, float, float]],
    *,
    total_s: float,
    horizons: tuple[int, ...] = HORIZONS_S,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(anchor_times, labels)`` with labels shaped ``(n_anchors, len(horizons))``."""
    stages = sleep_stage_epochs(events)
    stage_start = np.asarray([s for s, _, _ in stages])
    stage_end = np.asarray([e for _, e, _ in stages])
    stage_sleep = np.asarray([c in SLEEP_STAGES for _, _, c in stages])
    disturb = _disturbance_times(events)
    arousals = classify_arousals(events)
    onsets = np.asarray([a.onset for a in arousals])
    spontaneous = np.asarray([a.kind == "spontaneous" for a in arousals])

    t = np.arange(QUIET_S + EPOCH_S, total_s - max(horizons), STEP_S)
    # Stage of the last completed epoch: the stage run that contains t - EPOCH_S.
    i = np.searchsorted(stage_start, t - EPOCH_S, side="right") - 1
    in_run = (i >= 0) & (t - EPOCH_S < stage_end[np.clip(i, 0, None)])
    asleep = in_run & stage_sleep[np.clip(i, 0, None)]
    n_recent = np.searchsorted(disturb, t, side="right") - np.searchsorted(disturb, t - QUIET_S, side="left")
    t = t[asleep & (n_recent == 0)]

    labels = np.zeros((t.size, len(horizons)), dtype=np.int8)
    if onsets.size == 0:
        return t, labels
    nxt = np.searchsorted(onsets, t, side="right")  # first onset strictly after t
    has_next = nxt < onsets.size
    j = np.minimum(nxt, onsets.size - 1)
    gap = np.where(has_next, onsets[j] - t, np.inf)
    first_spont = has_next & spontaneous[j]
    for k, h in enumerate(horizons):
        hit = gap <= h
        labels[:, k] = np.where(hit, np.where(first_spont, 1, -1), 0)
    return t, labels


CARDIAC_WINDOWS_S = (10.0, 60.0)
CARDIAC_NAMES = tuple(
    f"{name}_{int(w)}s" for w in CARDIAC_WINDOWS_S for name in ("hr", "rmssd", "sdnn", "hr_slope", "n_beats")
) + ("hr_ratio_10_60",)


def cardiac_features(beat_times: np.ndarray, rr: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """RR statistics over windows ending at each anchor, from beats closed at or before t.

    NaN where a window has fewer than 3 intervals (signal dropout); the model sees it
    as missing rather than as a fake zero heart rate.
    """
    out = np.full((anchors.size, len(CARDIAC_NAMES)), np.nan, dtype=np.float32)
    hi = np.searchsorted(beat_times, anchors, side="right")
    col = 0
    mean_hr = {}
    for w in CARDIAC_WINDOWS_S:
        lo = np.searchsorted(beat_times, anchors - w, side="right")
        for i in range(anchors.size):
            seg = rr[lo[i] : hi[i]]
            if seg.size < 3:
                continue
            hr = 60.0 / seg
            out[i, col] = hr.mean()
            out[i, col + 1] = np.sqrt(np.mean(np.diff(seg) ** 2))
            out[i, col + 2] = seg.std()
            out[i, col + 3] = np.polyfit(beat_times[lo[i] : hi[i]] - anchors[i], hr, 1)[0]
            out[i, col + 4] = seg.size
        mean_hr[w] = out[:, col]
        col += 5
    out[:, col] = mean_hr[10.0] / mean_hr[60.0]
    return out


PRIOR_NAMES = ("since_last_arousal_s", "since_sleep_onset_s", "stage_n1", "stage_n2", "stage_n3", "stage_rem")


def prior_features(events: list[tuple[str, float, float]], anchors: np.ndarray) -> np.ndarray:
    """Context a clinician-free model could know at t: time since the last arousal
    onset, time since the first scored sleep epoch, and the last completed stage."""
    onsets = np.sort(np.asarray([s for c, s, _ in events if "arousal" in c.lower()], dtype=np.float64))
    stages = sleep_stage_epochs(events)
    sleep_onset = next((s for s, _, c in stages if c in SLEEP_STAGES), 0.0)
    k = np.searchsorted(onsets, anchors, side="right") - 1
    since = np.where(k >= 0, anchors - onsets[np.maximum(k, 0)] if onsets.size else np.inf, np.inf)
    since = np.minimum(since, 3600.0)  # ponytail: cap at 1 h; log/spline transform is the upgrade path
    out = np.zeros((anchors.size, len(PRIOR_NAMES)), dtype=np.float32)
    out[:, 0] = since
    out[:, 1] = anchors - sleep_onset
    stage_cols = {"Stage 1 sleep": 2, "Stage 2 sleep": 3, "Stage 3 sleep": 4, "Stage 4 sleep": 4, "REM sleep": 5}
    starts = np.asarray([s for s, _, _ in stages])
    idx = np.searchsorted(starts, anchors - EPOCH_S, side="right") - 1
    for i, j in enumerate(idx):
        if j >= 0 and stages[j][2] in stage_cols:
            out[i, stage_cols[stages[j][2]]] = 1.0
    return out


EEG_BANDS_HZ = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12), "sigma": (12, 16), "beta": (16, 30)}
EEG_NAMES = tuple(f"eeg_{b}_rel" for b in EEG_BANDS_HZ) + ("eeg_log_power",)


def eeg_features(eeg: np.ndarray, fs: float, anchors: np.ndarray, *, window_s: float = 30.0) -> np.ndarray:
    """Relative band power of the ``window_s`` of EEG ending at each anchor."""
    n = int(window_s * fs)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    masks = [(freqs >= lo) & (freqs < hi) for lo, hi in EEG_BANDS_HZ.values()]
    total_mask = (freqs >= 0.5) & (freqs < 30)
    out = np.full((anchors.size, len(EEG_NAMES)), np.nan, dtype=np.float32)
    ends = (anchors * fs).astype(np.int64)
    ok = (ends - n >= 0) & (ends <= eeg.size)
    if not ok.any():
        return out
    segs = np.stack([eeg[e - n : e] for e in ends[ok]])
    segs = segs - segs.mean(axis=1, keepdims=True)
    psd = np.abs(np.fft.rfft(segs * np.hanning(n), axis=1)) ** 2
    total = psd[:, total_mask].sum(axis=1) + 1e-20
    out[ok, : len(masks)] = np.stack([psd[:, m].sum(axis=1) / total for m in masks], axis=1)
    out[ok, -1] = np.log10(total)
    return out
