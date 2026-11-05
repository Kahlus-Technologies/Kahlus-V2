from __future__ import annotations

import unittest

import numpy as np

from neurotwin.forecastability.clas_replay import REFRACTORY_S, clicks_before_arousal, so_clicks


class ClasReplayTests(unittest.TestCase):
    def test_clicks_follow_slow_waves_only_when_allowed(self) -> None:
        fs = 100.0
        t = np.arange(0, 60, 1 / fs)
        eeg = 80 * np.sin(2 * np.pi * 0.8 * t)  # large 0.8 Hz slow oscillation
        allowed = np.zeros(60, dtype=bool)
        allowed[20:40] = True
        c = so_clicks(eeg, fs, allowed)
        self.assertTrue(c.size > 3)
        self.assertTrue(((c >= 20) & (c < 41)).all())
        self.assertTrue((np.diff(c) >= REFRACTORY_S - 1e-9).all())

    def test_causal(self) -> None:
        fs = 100.0
        t = np.arange(0, 60, 1 / fs)
        eeg = 80 * np.sin(2 * np.pi * 0.8 * t)
        allowed = np.ones(60, dtype=bool)
        a = so_clicks(eeg, fs, allowed)
        eeg2 = eeg.copy()
        eeg2[3000:] = 0
        b = so_clicks(eeg2, fs, allowed)
        np.testing.assert_allclose(a[a < 29.5], b[b < 29.5])

    def test_small_noise_triggers_nothing(self) -> None:
        eeg = 5 * np.random.default_rng(0).standard_normal(6000)
        self.assertEqual(so_clicks(eeg, 100.0, np.ones(60, dtype=bool)).size, 0)

    def test_clicks_before_arousal(self) -> None:
        got = clicks_before_arousal(np.array([10.0, 20.0, 30.0]), np.array([12.0, 40.0]), window_s=3.0)
        np.testing.assert_array_equal(got, [True, False, False])


if __name__ == "__main__":
    unittest.main()
