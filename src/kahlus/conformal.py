"""Split-conformal alarm thresholds (issue #89).

Goal: pick an alarm threshold on calibration subjects so the miss rate on new
subjects stays below a chosen alpha, and report what that costs in false alarms.

Guarantee: if calibration and test positives are exchangeable, raising an alarm
when score >= ``miss_rate_threshold(cal_pos_scores, alpha)`` misses at most alpha
of test positives in expectation. Positives from one night are not independent,
so the guarantee is approximate; the per-subject coverage spread reported by the
caller is the honest check.
"""
from __future__ import annotations

import numpy as np


def miss_rate_threshold(cal_pos_scores: np.ndarray, alpha: float) -> float:
    """Largest threshold whose conformal miss-rate bound is <= alpha.

    With n calibration positives sorted ascending, alarming at the k-th smallest
    score misses at most (k - 1 + 1) / (n + 1) = k / (n + 1) of a new positive.
    """
    s = np.sort(np.asarray(cal_pos_scores, dtype=np.float64))
    n = s.size
    if n == 0:
        raise ValueError("need calibration positives")
    k = int(np.floor(alpha * (n + 1)))
    return float(s[k - 1]) if k >= 1 else -np.inf


def alarm_metrics(scores: np.ndarray, y: np.ndarray, threshold: float, *, step_s: float) -> dict:
    """Miss rate on positives and false alarms per hour of anchors."""
    alarm = scores >= threshold
    pos = y == 1
    hours = y.size * step_s / 3600.0
    return {
        "miss_rate": float(1.0 - alarm[pos].mean()) if pos.any() else float("nan"),
        "false_alarms_per_hour": float((alarm & ~pos).sum() / hours) if hours else float("nan"),
        "alarm_fraction": float(alarm.mean()),
    }
