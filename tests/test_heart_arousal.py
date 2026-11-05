from __future__ import annotations

import unittest

import numpy as np

from neurotwin.forecastability.heart_arousal import anchor_labels

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


if __name__ == "__main__":
    unittest.main()
