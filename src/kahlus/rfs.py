"""Residual forecastability score (RFS): bits of predictive information a model adds over a baseline.

Moved from the v1 `neurotwin.forecastability.m1` module (Kahlus-V1) unchanged in behavior.
"""

from __future__ import annotations

import numpy as np


def _loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    y = np.asarray(y, dtype=np.float64)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def nll(y: np.ndarray, p: np.ndarray) -> float:
    """Mean binary negative log-likelihood in nats, probabilities clipped to [1e-5, 1 - 1e-5]."""
    return float(np.mean(_loss(y, p)))


def rfs_bits(y: np.ndarray, baseline: np.ndarray, pred: np.ndarray) -> float:
    """Baseline NLL minus model NLL, in bits. Positive means the model predicts better."""
    return float((nll(y, baseline) - nll(y, pred)) / np.log(2.0))


def cluster_bootstrap_rfs(
    y: np.ndarray,
    baseline: np.ndarray,
    pred: np.ndarray,
    subject: np.ndarray,
    *,
    seed: int,
    mode: str = "smoke",
) -> dict[str, float | str]:
    """95% percentile CI for rfs_bits, resampling whole subjects. smoke: 300 resamples, claim: 2000."""
    n_boot = 2_000 if mode == "claim" else 300
    _, inv = np.unique(subject, return_inverse=True)
    # RFS is a ratio of sums, so resampling whole subjects needs only each subject's summed
    # log-loss difference and window count, not the windows themselves.
    n = np.bincount(inv).astype(np.float64)
    diff = np.bincount(inv, weights=_loss(y, baseline) - _loss(y, pred))
    pick = np.random.default_rng(seed).integers(0, n.size, size=(n_boot, n.size))
    samples = diff[pick].sum(1) / n[pick].sum(1) / np.log(2.0)
    return {
        "rfs_ci_low": float(np.percentile(samples, 2.5)),
        "rfs_ci_high": float(np.percentile(samples, 97.5)),
        "bootstrap_mode": mode,
        "bootstrap_samples": n_boot,
    }
