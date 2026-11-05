from __future__ import annotations

import unittest

import numpy as np

from neurotwin.forecastability.heart_arousal import CAUSAL_WARMUP, causal_z, half_night_shift


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


class HalfNightShiftTests(unittest.TestCase):
    def test_is_a_pure_time_permutation(self) -> None:
        x = np.arange(10.0)[:, None]
        np.testing.assert_array_equal(half_night_shift(x)[:, 0], [5, 6, 7, 8, 9, 0, 1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
