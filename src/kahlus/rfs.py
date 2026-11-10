"""Residual forecastability score (RFS): bits of predictive information a model adds over a baseline.

Moved from the v1 `neurotwin.forecastability.m1` module (Kahlus-V1) unchanged in behavior.
"""

from __future__ import annotations

import numpy as np


def nll(y: np.ndarray, p: np.ndarray) -> float:
    """Mean binary negative log-likelihood in nats, probabilities clipped to [1e-5, 1 - 1e-5]."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    y = np.asarray(y, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


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
    rng = np.random.default_rng(seed)
    unique = np.unique(subject)
    samples = []
    for _ in range(n_boot):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(subject == group) for group in chosen])
        samples.append(rfs_bits(y[idx], baseline[idx], pred[idx]))
    return {
        "rfs_ci_low": float(np.percentile(samples, 2.5)),
        "rfs_ci_high": float(np.percentile(samples, 97.5)),
        "bootstrap_mode": mode,
        "bootstrap_samples": n_boot,
    }
