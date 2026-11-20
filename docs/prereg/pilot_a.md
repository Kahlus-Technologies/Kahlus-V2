# Pilot A (partial data, not a claim): pretrained 10M encoder vs the same architecture from scratch

Written 2026-11-20 before any run. Data: the SHHS-1 shards on Engram at the time of the run (count in the report),
dev nights 200001-200025 excluded. Encoder: /shared/aayu/fill-a/runs/kahlus-10m/ckpt-00003139.pt (3,139 steps).
Task: kahlus.heart_arousal labels, spontaneous arousal, anchors as in K1. Window: 64 s of C4 + ECG ending at least 2 s
before the anchor. Splits: test = is_heldout subjects (never seen in pretraining), tune = is_tuning, fit = the rest.
Seeds 0, 1, 2 for every arm.

Hypothesis (two-sided): fine-tuning the pretrained encoder forecasts better than fine-tuning the same architecture
from random weights, at the same steps, batch and learning-rate grid.

Primary metric: bits over the prior-only logistic on test subjects, at horizon 30 s. Primary cell: fine-tune,
pretrained minus scratch, mean over seeds per subject, paired subject bootstrap.

Secondary (Holm over the 8 cells): horizons 5, 15, 60, 120 s for the fine-tune contrast; linear probe pretrained
minus random-init at 30 s; linear probe pretrained minus prior + band power (heart_arousal.eeg_features) at 30 s;
linear probe pretrained with per-window renormalised input minus as-is at 30 s.

Controls: (1) shift control: the pretrained fine-tune on half-night-shifted windows must give bits whose interval
includes 0 at 30 s; if not, the window carries identity or time-of-night, not arousal timing, and no A result is
reported until explained. (2) Seed spread: report each seed.

Gate for 50M/200M ("clear gain", captain call 3A): the primary interval lies above 0 AND the gain is at least
+0.002 bits AND control 1 passes. Kill rule: otherwise the larger sweep stays gated; this pilot is reported as a
null or unresolved for this checkpoint only, and the confirmation on full data uses the same text.
