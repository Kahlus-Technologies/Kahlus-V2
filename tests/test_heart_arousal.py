from __future__ import annotations

import unittest

import numpy as np

from neurotwin.forecastability.heart_arousal import (
    BEAT_NAMES,
    CARDIAC_NAMES,
    EEG_NAMES,
    PRIOR_NAMES,
    anchor_labels,
    beat_features,
    cardiac_features,
    eeg_features,
    prior_features,
)

NIGHT = [("Stage 2 sleep", 0, 3000)]


class AnchorLabelTests(unittest.TestCase):
    def test_label_by_first_arousal_kind(self) -> None:
        events = NIGHT + [("Arousal", 1000, 5), ("Hypopnea", 2000, 20), ("Arousal", 2021, 5)]
        t, y = anchor_labels(events, total_s=3000, horizons=(5, 30))
        row = {float(a): tuple(r) for a, r in zip(t, y)}
        self.assertEqual(row[995.0], (1, 1))  # spontaneous onset 5 s ahead
        self.assertEqual(row[980.0], (0, 1))  # 20 s ahead: outside 5 s, inside 30 s
        self.assertEqual(row[1900.0], (0, 0))
        self.assertEqual(row[1995.0], (0, -1))  # hypopnea at 2000 is still future; its arousal is excluded
        self.assertNotIn(2025.0, row)  # hypopnea started 25 s ago, inside the quiet window

    def test_respiratory_first_arousal_is_excluded(self) -> None:
        events = NIGHT + [("Hypopnea", 1500, 10), ("Arousal", 1512, 5)]
        t, y = anchor_labels(events, total_s=3000, horizons=(30,))
        self.assertEqual(dict(zip(t.tolist(), y[:, 0].tolist()))[1490.0], -1)

    def test_no_anchor_within_quiet_window_after_disturbance(self) -> None:
        events = NIGHT + [("Arousal", 1000, 5)]
        t, _ = anchor_labels(events, total_s=3000, horizons=(5,))
        self.assertFalse(np.any((t >= 1000) & (t < 1060)))
        self.assertIn(1065.0, t.tolist())

    def test_current_epoch_score_is_not_used(self) -> None:
        # Epoch 1500-1530 is scored Wake (the scorer saw the future). An anchor at
        # 1505 must still exist because the last completed epoch (1470-1500) was sleep.
        events = [("Stage 2 sleep", 0, 1500), ("Wake", 1500, 30), ("Stage 2 sleep", 1530, 1470)]
        t, _ = anchor_labels(events, total_s=3000, horizons=(5,))
        self.assertIn(1505.0, t.tolist())
        self.assertNotIn(1535.0, t.tolist())  # wake started 35 s ago, inside quiet window


class FeatureTests(unittest.TestCase):
    def test_cardiac_values_and_causality(self) -> None:
        beats = np.arange(0.0, 200.0, 1.0)  # 60 bpm
        rr = np.diff(beats)
        times = beats[1:]
        anchors = np.array([100.0])
        f = cardiac_features(times, rr, anchors)[0]
        named = dict(zip(CARDIAC_NAMES, f))
        self.assertAlmostEqual(named["hr_60s"], 60.0, places=4)
        self.assertAlmostEqual(named["rmssd_60s"], 0.0, places=6)
        self.assertAlmostEqual(named["hr_ratio_10_60"], 1.0, places=4)
        # Beats after the anchor must not change anything.
        fast = np.concatenate([beats[beats <= 100], 100 + np.cumsum(np.full(100, 0.4))])
        g = cardiac_features(fast[1:], np.diff(fast), anchors)[0]
        np.testing.assert_allclose(f, g)

    def test_beat_features_are_relative_and_causal(self) -> None:
        beats = np.arange(0.0, 100.0, 1.0)
        beats = np.concatenate([beats, 99 + np.cumsum([0.8, 0.8])])  # two faster beats end at 100.6
        times, rr = beats[1:], np.diff(beats)
        f = dict(zip(BEAT_NAMES, beat_features(times, rr, np.array([100.7]))[0]))
        self.assertAlmostEqual(f["rr_last1_rel"], 0.8, places=5)
        self.assertAlmostEqual(f["rr_last3_rel"], 1.0, places=5)
        self.assertAlmostEqual(f["since_last_beat_s"], 0.1, places=5)
        later = np.concatenate([beats, [101.0, 101.2]])
        g = beat_features(later[1:], np.diff(later), np.array([100.7]))[0]
        np.testing.assert_allclose(list(f.values()), g)

    def test_cardiac_dropout_is_nan(self) -> None:
        f = cardiac_features(np.array([1.0, 2.0]), np.array([1.0, 1.0]), np.array([100.0]))
        self.assertTrue(np.isnan(f).all())

    def test_prior_features(self) -> None:
        events = NIGHT + [("Arousal", 500, 5)]
        f = prior_features(events, np.array([400.0, 700.0]))
        named = [dict(zip(PRIOR_NAMES, row)) for row in f]
        self.assertEqual(named[0]["since_last_arousal_s"], 3600.0)  # none yet, capped
        self.assertEqual(named[1]["since_last_arousal_s"], 200.0)
        self.assertEqual(named[1]["stage_n2"], 1.0)

    def test_eeg_band_and_causality(self) -> None:
        fs = 128.0
        t = np.arange(int(fs * 200)) / fs
        eeg = np.sin(2 * np.pi * 10 * t)
        f = eeg_features(eeg, fs, np.array([100.0]))[0]
        named = dict(zip(EEG_NAMES, f))
        self.assertGreater(named["eeg_alpha_rel"], 0.9)
        eeg2 = eeg.copy()
        eeg2[int(100 * fs) :] = np.random.default_rng(0).standard_normal(eeg2.size - int(100 * fs))
        np.testing.assert_allclose(f, eeg_features(eeg2, fs, np.array([100.0]))[0])


if __name__ == "__main__":
    unittest.main()
