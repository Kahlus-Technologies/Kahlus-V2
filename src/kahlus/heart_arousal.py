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

from kahlus.nsrr import SLEEP_STAGES, classify_arousals, event_kind, sleep_stage_epochs

HORIZONS_S = (5, 15, 30, 60, 120)
STEP_S = 5.0
QUIET_S = 60.0
EPOCH_S = 30.0


def _disturbance_intervals(events: list[tuple[str, float, float]]) -> tuple[np.ndarray, np.ndarray]:
    """(start, end) of every disturbance, sorted by start. Arousal, respiratory and
    limb events count for their whole scored duration. A Wake run counts from the
    end of its first 30 s epoch (when it becomes knowable) to the end of the run;
    NSRR merges consecutive same-stage epochs into one event."""
    spans = []
    for concept, start, dur in events:
        if concept == "Wake":
            spans.append((start + EPOCH_S, start + max(dur, EPOCH_S)))
        elif event_kind(concept) is not None:
            spans.append((start, start + dur))
    spans.sort()
    return np.asarray([s for s, _ in spans], dtype=np.float64), np.asarray([e for _, e in spans], dtype=np.float64)


def anchor_labels(
    events: list[tuple[str, float, float]],
    *,
    total_s: float,
    horizons: tuple[int, ...] = HORIZONS_S,
    step_s: float = STEP_S,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(anchor_times, labels)`` with labels shaped ``(n_anchors, len(horizons))``."""
    stages = sleep_stage_epochs(events)
    if not stages:  # unscored night: no stage is ever known, so no anchor qualifies
        return np.zeros(0), np.zeros((0, len(horizons)), dtype=np.int8)
    stage_start = np.asarray([s for s, _, _ in stages])
    stage_end = np.asarray([e for _, e, _ in stages])
    stage_sleep = np.asarray([c in SLEEP_STAGES for _, _, c in stages])
    d_start, d_end = _disturbance_intervals(events)
    arousals = classify_arousals(events)
    onsets = np.asarray([a.onset for a in arousals])
    spontaneous = np.asarray([a.kind == "spontaneous" for a in arousals])

    t = np.arange(QUIET_S + EPOCH_S, total_s - max(horizons), step_s)
    # Stage of the last completed epoch: the stage run that contains t - EPOCH_S.
    i = np.searchsorted(stage_start, t - EPOCH_S, side="right") - 1
    in_run = (i >= 0) & (t - EPOCH_S < stage_end[np.clip(i, 0, None)])
    asleep = in_run & stage_sleep[np.clip(i, 0, None)]
    # Quiet = no disturbance overlapping [t - QUIET_S, t]: among events started by t,
    # the latest end must be at or before t - QUIET_S.
    started = np.searchsorted(d_start, t, side="right")
    latest_end = np.concatenate([[-np.inf], np.maximum.accumulate(d_end)])[started]
    t = t[asleep & (latest_end <= t - QUIET_S)]

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

LAST_BEATS = 5
BEAT_NAMES = tuple(f"rr_last{i}_rel" for i in range(1, LAST_BEATS + 1)) + ("since_last_beat_s",)


def beat_features(beat_times: np.ndarray, rr: np.ndarray, anchors: np.ndarray, *, baseline_s: float = 60.0) -> np.ndarray:
    """The last LAST_BEATS RR intervals closed at or before t, each divided by the
    median RR of the preceding ``baseline_s``, plus time since the last beat.
    No interpolation, so nothing after t leaks in. Ratios below 1 mean speeding up."""
    out = np.full((anchors.size, len(BEAT_NAMES)), np.nan, dtype=np.float32)
    hi = np.searchsorted(beat_times, anchors, side="right")
    lo = np.searchsorted(beat_times, anchors - baseline_s, side="right")
    for i in range(anchors.size):
        if hi[i] - lo[i] < LAST_BEATS + 3:
            continue
        base = np.median(rr[lo[i] : hi[i]])
        out[i, :LAST_BEATS] = rr[hi[i] - LAST_BEATS : hi[i]][::-1] / base
        out[i, -1] = anchors[i] - beat_times[hi[i] - 1]
    return out


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
    short, long = CARDIAC_WINDOWS_S
    out[:, col] = mean_hr[short] / mean_hr[long]
    return out


PRIOR_NAMES = ("since_last_arousal_s", "since_sleep_onset_s", "stage_n1", "stage_n2", "stage_n3", "stage_rem")


def prior_features(events: list[tuple[str, float, float]], anchors: np.ndarray) -> np.ndarray:
    """Context a clinician-free model could know at t: time since the last arousal
    onset, time since the first scored sleep epoch, and the last completed stage."""
    onsets = np.sort(np.asarray([s for c, s, _ in events if event_kind(c) == "arousal"], dtype=np.float64))
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
        if j >= 0 and anchors[i] - EPOCH_S < stages[j][1] and stages[j][2] in stage_cols:
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


def build_recording(edf_path, xml_path, *, ecg_labels: tuple[str, ...], eeg_labels: tuple[str, ...], ppg_labels: tuple[str, ...] = ()) -> dict:
    """Anchors, labels and every feature block for one night."""
    from kahlus.nsrr import load_nsrr_epoch_matrix, parse_nsrr_events
    from kahlus.qrs_fast import causal_beats

    groups = {"ecg": ecg_labels, "eeg": eeg_labels}
    if ppg_labels:
        groups["ppg"] = ppg_labels
    m = load_nsrr_epoch_matrix(edf_path, channel_groups=groups)
    fs = m["sfreq"]
    flat = {g: m["epochs"][g][:, :, 0].ravel() for g in m["epochs"]}
    events = parse_nsrr_events(xml_path)
    anchors, labels = anchor_labels(events, total_s=flat["ecg"].size / fs)
    beat_times, rr = causal_beats(flat["ecg"], fs)
    blocks = {
        "cardiac": cardiac_features(beat_times, rr, anchors),
        "prior": prior_features(events, anchors),
        "eeg": eeg_features(flat["eeg"], fs, anchors),
    }
    if "ppg" in flat:
        # ponytail: PPG treated as a pulse wave through the same peak detector; a
        # dedicated PPG beat detector (and pulse arrival time vs ECG) is the upgrade path.
        pt, pp = causal_beats(flat["ppg"], fs)
        blocks["ppg"] = cardiac_features(pt, pp, anchors)
    return {"anchors": anchors, "labels": labels, "blocks": blocks, "n_beats": int(rr.size)}


# Shared by every forecasting script (benchmark, beat warning, conformal, CLAS, montage).

#: EDF channel labels per NSRR cohort. C4 EEG is the derivation both cohorts share.
COHORTS = {
    "mesa": {"ecg_labels": ("EKG",), "eeg_labels": ("EEG3",), "ppg_labels": ("Pleth",)},
    "shhs": {"ecg_labels": ("ECG",), "eeg_labels": ("EEG",), "ppg_labels": ()},
}

CAUSAL_WARMUP = 60  # anchors (5 min at 5 s spacing) before running stats are trusted


def causal_z(x: np.ndarray) -> np.ndarray:
    """Z-score each row against the mean and SD of earlier rows in the same night
    only (expanding window, NaN-aware). Rows before CAUSAL_WARMUP become NaN.
    Anchors are time-ordered, so row i never sees row i or later."""
    ok = ~np.isnan(x)
    v = np.where(ok, x, 0.0).astype(np.float64)
    n = np.cumsum(ok, axis=0) - ok  # count of earlier non-NaN rows
    s1 = np.cumsum(v, axis=0) - v
    s2 = np.cumsum(v * v, axis=0) - v * v
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = s1 / n
        sd = np.sqrt(np.maximum(s2 / n - mu * mu, 0.0))
        z = (x - mu) / np.where(sd > 0, sd, np.nan)
    z[n < CAUSAL_WARMUP] = np.nan
    return z.astype(np.float32)


def half_night_shift(x: np.ndarray) -> np.ndarray:
    """Falsification control: rotate one night's rows by half its length. Keeps the
    night's own values, drift and missingness but breaks alignment with labels."""
    return np.roll(x, x.shape[0] // 2, axis=0)


def make_model(seed: int):
    """The one model used across all forecasting comparisons; NaN-tolerant."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    # ponytail: one fixed gradient-boosting config; a tuned or sequence model is the upgrade path.
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=seed)


def fit_predict(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, seed: int) -> np.ndarray:
    return make_model(seed).fit(x_tr, y_tr).predict_proba(x_te)[:, 1]
