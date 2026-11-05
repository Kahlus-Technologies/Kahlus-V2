from __future__ import annotations

import unittest

import numpy as np

from neurotwin.forecastability.conformal import alarm_metrics, miss_rate_threshold


class ConformalTests(unittest.TestCase):
    def test_threshold_controls_miss_rate_on_exchangeable_data(self) -> None:
        rng = np.random.default_rng(0)
        misses = []
        for _ in range(400):
            cal = rng.normal(size=200)
            test = rng.normal(size=200)
            thr = miss_rate_threshold(cal, alpha=0.1)
            misses.append(np.mean(test < thr))
        self.assertLessEqual(np.mean(misses), 0.1 + 0.005)
        self.assertGreater(np.mean(misses), 0.08)  # not trivially conservative

    def test_tiny_alpha_alarms_on_everything(self) -> None:
        self.assertEqual(miss_rate_threshold(np.array([0.2, 0.5]), alpha=0.1), -np.inf)

    def test_alarm_metrics(self) -> None:
        y = np.array([1, 0, 0, 1, 0, 0])
        s = np.array([0.9, 0.8, 0.1, 0.2, 0.1, 0.1])
        m = alarm_metrics(s, y, 0.5, step_s=600.0)  # 6 anchors x 10 min = 1 h
        self.assertEqual(m["miss_rate"], 0.5)
        self.assertEqual(m["false_alarms_per_hour"], 1.0)


if __name__ == "__main__":
    unittest.main()
