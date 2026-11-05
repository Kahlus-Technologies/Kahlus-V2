from __future__ import annotations

import unittest

import numpy as np

from neurotwin.adapters.qrs_fast import beat_intervals, detect_qrs, good_beat_mask, heart_rate_grid


def synthetic_ecg(fs: float, seconds: float, bpm: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian QRS spikes plus a broad T-wave and noise. Returns (ecg, true beat samples)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * seconds)) / fs
    beats = np.arange(0.5, seconds - 0.5, 60.0 / bpm)
    ecg = 0.02 * rng.standard_normal(t.size)
    for b in beats:
        ecg += np.exp(-0.5 * ((t - b) / 0.01) ** 2)
        ecg += 0.3 * np.exp(-0.5 * ((t - b - 0.25) / 0.05) ** 2)
    return ecg, np.round(beats * fs).astype(int)


class QrsTests(unittest.TestCase):
    def test_detects_every_beat_and_skips_t_waves(self) -> None:
        for fs in (125.0, 256.0):
            ecg, truth = synthetic_ecg(fs, 120, bpm=62)
            peaks = detect_qrs(ecg, fs)
            self.assertEqual(len(peaks), len(truth), fs)
            self.assertLess(np.max(np.abs(peaks - truth)), int(0.05 * fs))

    def test_beat_intervals_drop_nonphysiologic(self) -> None:
        fs = 100.0
        peaks = np.array([0, 100, 200, 210, 310, 1310])  # 10 ms (600 bpm) and 10 s (6 bpm) gaps
        times, rr = beat_intervals(peaks, fs)
        np.testing.assert_allclose(rr, [1.0, 1.0, 1.0])
        np.testing.assert_allclose(times, [1.0, 2.0, 3.1])

    def test_heart_rate_grid_matches_truth(self) -> None:
        ecg, _ = synthetic_ecg(125.0, 300, bpm=70)
        hr, info = heart_rate_grid(ecg, 125.0, 300)
        self.assertIsNotNone(hr)
        self.assertAlmostEqual(info["median_bpm"], 70.0, delta=1.0)

    def test_good_beat_mask_drops_double_counted_block(self) -> None:
        normal = np.arange(0.0, 600.0, 1.0)  # 60 bpm for 10 min
        doubled = np.arange(600.0, 900.0, 0.4)  # 150 bpm for 5 min
        burst = np.arange(900.0, 1500.0, 1.0)
        burst[100:110] = burst[100] + np.arange(10) * 0.45  # 10-beat tachy burst, keep
        beats = np.sort(np.unique(np.concatenate([normal, doubled, burst])))
        times, rr = beats[1:], np.diff(beats)
        good = good_beat_mask(times, rr)
        self.assertTrue(good[times < 600].all())
        self.assertFalse(good[(times >= 600) & (times < 900)].any())
        self.assertTrue(good[times >= 900].all())


if __name__ == "__main__":
    unittest.main()
