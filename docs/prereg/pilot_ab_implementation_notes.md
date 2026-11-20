# Pilots A and B: where the code departs from the plan (written before any run)

`docs/prereg/pilot_a.md` and `pilot_b.md` are the planner's texts, unchanged. These notes list the points where
`scripts/pilot_ab.py` had to differ from the planner's runbook. None changes a hypothesis, a metric or a gate.

1. **`encode`, not `embed`.** `KahlusPatchModel.embed` already names the input projection. The new method is `encode`, on top of a new `hidden`.
2. **Window offset.** Stored patch `j` covers second `[j + 60, j + 61)`, not `[j + 59, j + 60)`: `causal_patches` drops 59 z-score
   rows after the warm-up row, so stored `j` is raw patch `j + 60`. A window ends at `j_end = floor(t) - 63`, which keeps the 2 s margin
   (asserted in `window_end`). This is one second more conservative than the plan's `floor(t) - 62`.
3. **File formats.** `anchors.npz` and `bwin.npz` replace `anchors.parquet` (no pandas on the node).
4. **Pilot B windows.** Pilot B uses one window ending at every 30 s epoch boundary (from 90 s), not also the Pilot A anchor windows. The
   subject-ID probe trains on at most about 100 windows per subject.
5. **Band power.** The Pilot A band-power baseline is computed from the stored causal-z-scored C4 patches (relative power in the five
   `heart_arousal.EEG_BANDS_HZ` bands, and log power of the z-scored signal), not from the raw EDF, so it needs no EDF read. Pilot B's
   amplitude is the raw C4 RMS in microvolts from the EDF, as planned.
6. **Probes.** Linear probes run at 30 s only (every secondary probe cell is at 30 s), on at most 100,000 random fit and 100,000 tune
   anchors. The prior-only logistic runs at every horizon.
7. **Fine-tune.** Each run takes minutes, so a run resumes at run level (a finished run leaves `pred.npz`), not by a checkpoint every 15
   minutes. The 4,000 fit anchors per step are drawn with replacement; both arms of a seed see the same draw sequence.
8. **Secondary cells.** There are 7 (four horizons of the fine-tune contrast, three probe contrasts), and Holm runs over 7, not 8.
9. **Pilot B findings.** "Falls by half" for the amplitude probe means `(per-window acc - 0.2) <= 0.5 * (as-is acc - 0.2)`. Amplitude and
   stage probes use the first 80% of the 200 subjects (by ID) for fitting and the last 20% for testing.
10. **Shift control.** The shifted window is the window of the same night half a night later, wrapping around, never before second 63.
11. **Pretrained embeddings** are deterministic, so there is one pretrained embedding file (seed 0) per input.
12. **Data size.** The kept shards hold 1,902 nights, not 873 (the build log says so). The report prints the exact counts.
