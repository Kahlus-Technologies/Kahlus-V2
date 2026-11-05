from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from run_heart_arousal_benchmark import CAUSAL_WARMUP, causal_z  # noqa: E402


class CausalZTests(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.random.default_rng(0).normal(5, 2, (200, 2)).astype(np.float32)
        self.x[150, 1] = np.nan

    def test_matches_stats_of_earlier_rows_only(self) -> None:
        z = causal_z(self.x)
        i = 100
        ref = (self.x[i] - self.x[:i].mean(0)) / self.x[:i].std(0)
        np.testing.assert_allclose(z[i], ref, atol=1e-4)

    def test_future_rows_do_not_matter(self) -> None:
        x2 = self.x.copy()
        x2[101:] = 999.0
        np.testing.assert_allclose(causal_z(x2)[100], causal_z(self.x)[100])

    def test_warmup_and_nan_handling(self) -> None:
        z = causal_z(self.x)
        self.assertTrue(np.isnan(z[:CAUSAL_WARMUP]).all())
        self.assertTrue(np.isnan(z[150, 1]))
        self.assertFalse(np.isnan(z[151, 1]))


if __name__ == "__main__":
    unittest.main()
