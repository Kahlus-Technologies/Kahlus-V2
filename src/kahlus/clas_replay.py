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


def so_clicks(eeg_uv: np.ndarray, fs: float, allowed: np.ndarray) -> np.ndarray:
    """Click times (s). ``allowed`` is a boolean per-second mask (N2/N3 known at t)."""
    b, a = butter(2, [0.5 / (fs / 2), 4.0 / (fs / 2)], btype="band")
    x = lfilter(b, a, eeg_uv)
    below = x < NEG_UV
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
