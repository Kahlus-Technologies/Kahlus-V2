from __future__ import annotations

import unittest

import numpy as np

from neurotwin.forecastability.montage_degrade import LEVELS, add_white_noise, degrade, downsample


class MontageDegradeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fs = 128.0
        self.x = 30 * np.sin(2 * np.pi * 10 * np.arange(int(self.fs * 120)) / self.fs)

    def test_white_noise_rms(self) -> None:
        noise = add_white_noise(np.zeros(100_000), 7.0)
        self.assertAlmostEqual(noise.std(), 7.0, delta=0.1)

    def test_levels_are_reproducible_and_ordered(self) -> None:
        err = {}
        for level in LEVELS:
            a, b = degrade(self.x, self.fs, level, seed=3), degrade(self.x, self.fs, level, seed=3)
            np.testing.assert_array_equal(a, b)
            err[level] = float(np.std(a - self.x))
        self.assertEqual(err["clean"], 0.0)
        self.assertLess(err["noise_5uv"], err["noise_10uv"])
        self.assertLess(err["noise_10uv"], err["noise_20uv"])
        self.assertLess(err["dry_headband"], err["dry_headband_bad"])

    def test_downsample_keeps_alpha(self) -> None:
        y, fs2 = downsample(self.x, self.fs, 100.0)
        self.assertAlmostEqual(fs2, 100.0, places=6)
        f = np.fft.rfftfreq(y.size, 1 / fs2)
        self.assertAlmostEqual(f[np.argmax(np.abs(np.fft.rfft(y)))], 10.0, delta=0.1)


if __name__ == "__main__":
    unittest.main()
