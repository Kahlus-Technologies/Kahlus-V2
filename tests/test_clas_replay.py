from __future__ import annotations

import unittest

import numpy as np

from kahlus.clas_replay import (
    GUARD_S,
    MEAN_DIFFERENCE,
    RELATIVE_RISK,
    REFRACTORY_S,
    clicks_before_arousal,
    first_of_runs,
    hr_change,
    matched_controls,
    rotate_clicks,
    running_level,
    since_arousal_bin,
    so_clicks,
    subject_bootstrap,
)


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

    def test_matched_controls_follow_the_preregistered_rule(self) -> None:
        stage = np.zeros(3000, dtype=np.int8)
        stage[100:1500] = 2
        stage[1500:2500] = 3
        bins = since_arousal_bin(np.array([50.0, 1000.0]), 3000)
        clicks = np.array([300.0, 700.5, 1200.0, 1800.0, 2900.0])  # the last is outside any allowed stretch
        got, dropped = matched_controls(clicks, stage, bins, rng=np.random.default_rng(0))
        self.assertEqual(dropped, 1)
        self.assertTrue(np.isnan(got[4]).all())
        for i in range(4):
            c = got[i][np.isfinite(got[i])]
            self.assertEqual(c.size, 5)
            self.assertTrue((stage[c.astype(int)] == stage[int(clicks[i])]).all())
            self.assertTrue((bins[c.astype(int)] == bins[int(clicks[i])]).all())
            self.assertTrue((np.abs(c - clicks[i]) <= 600).all())
            self.assertTrue((np.abs(c[:, None] - clicks[None, :4]).min(1) >= GUARD_S).all())
            self.assertTrue((np.abs(c[:, None] - c[None, :]) + 99 * np.eye(5) >= GUARD_S).all())

    def test_since_arousal_bins(self) -> None:
        got = since_arousal_bin(np.array([100.0]), 1000)
        self.assertEqual((got[50], got[200], got[300], got[800]), (2, 0, 1, 2))

    def test_rotated_clicks_avoid_real_clicks_and_closed_trigger(self) -> None:
        stage = np.full(100, 2, dtype=np.int8)
        stage[:10] = 0
        got = rotate_clicks(np.array([5.0, 60.0, 80.0]), stage)  # rotated: 55, 10, 30
        np.testing.assert_allclose(got, [10.0, 30.0, 55.0])
        stage[50:] = 0
        np.testing.assert_allclose(rotate_clicks(np.array([5.0, 60.0, 80.0]), stage), [10.0, 30.0])

    def test_first_of_runs_and_running_level(self) -> None:
        np.testing.assert_allclose(first_of_runs(np.array([10.0, 12.5, 15.0, 40.0, 42.5])), [10.0, 40.0])
        x = np.concatenate([np.full(100, 10.0), np.full(100, 100.0)])
        lvl = running_level(x, 1.0, pct=90.0)
        self.assertEqual(lvl[0], -40.0)  # before any sample: the default level
        self.assertEqual(lvl[50], -10.0)
        self.assertTrue(lvl[150] <= -10.0)  # follows the amplitude seen so far, never the future

    def test_hr_change_compares_after_with_before(self) -> None:
        bt = np.arange(0.0, 60.0, 1.0)
        rr = np.where(bt > 33, 0.5, 1.0)  # 60 bpm, then 120 bpm from t = 34
        got = hr_change(bt, rr, np.array([30.0, 5.0, 55.0, 1.0]))
        self.assertAlmostEqual(got[0], 112.5 - 60.0)  # beats 33..40 after, 21..30 before
        self.assertEqual(got[1], 0.0)
        self.assertTrue(np.isnan(got[2]))  # fewer than 3 beats after
        self.assertTrue(np.isnan(got[3]))  # fewer than 3 beats before

    def test_bootstrap_interval_brackets_the_estimate(self) -> None:
        rng = np.random.default_rng(0)
        n = rng.integers(80, 120, size=40)
        c = np.stack([rng.binomial(n, 0.2), rng.binomial(n, 0.1), n], axis=1).astype(float)
        r = subject_bootstrap(c, RELATIVE_RISK, n_boot=500, seed=1)
        self.assertAlmostEqual(r["estimate"], c[:, 0].sum() / c[:, 1].sum())
        self.assertTrue(r["ci_low"] < r["estimate"] < r["ci_high"] and r["p"] < 0.05)
        null = np.stack([rng.binomial(n, 0.1), rng.binomial(n, 0.1), n], axis=1).astype(float)
        self.assertTrue(subject_bootstrap(null, MEAN_DIFFERENCE, n_boot=500)["p"] > 0.05)


if __name__ == "__main__":
    unittest.main()
