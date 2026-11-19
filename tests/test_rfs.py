from __future__ import annotations

import unittest

import numpy as np

from kahlus.rfs import cluster_bootstrap_rfs, rfs_bits


def old_cluster_bootstrap(y, baseline, pred, subject, *, seed, n_boot):
    """The pre-fix path: rebuild every resampled subject's window index with a full scan."""
    rng = np.random.default_rng(seed)
    unique = np.unique(subject)
    samples = []
    for _ in range(n_boot):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(subject == group) for group in chosen])
        samples.append(rfs_bits(y[idx], baseline[idx], pred[idx]))
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


class ClusterBootstrapTests(unittest.TestCase):
    def test_fast_path_equals_old_path(self) -> None:
        rng = np.random.default_rng(1)
        subject = rng.choice(["a", "b", "c", "d", "e", "f", "g"], size=400, p=[.3, .2, .15, .1, .1, .1, .05])
        y = (rng.random(400) < 0.2).astype(int)
        base = np.clip(rng.random(400) * 0.4, 0, 1)
        pred = np.where(y == 1, 0.5, 0.1) * 0.7 + 0.3 * base  # includes clipped extremes below
        pred[:5], base[:5] = 0.0, 1.0
        got = cluster_bootstrap_rfs(y, base, pred, subject, seed=7, mode="smoke")
        low, high = old_cluster_bootstrap(y, base, pred, subject, seed=7, n_boot=300)
        self.assertAlmostEqual(got["rfs_ci_low"], low, places=10)
        self.assertAlmostEqual(got["rfs_ci_high"], high, places=10)


if __name__ == "__main__":
    unittest.main()
