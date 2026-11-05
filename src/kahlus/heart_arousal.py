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
