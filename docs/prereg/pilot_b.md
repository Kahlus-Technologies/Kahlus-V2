# Pilot B (partial data, not a claim): do the 10M embeddings encode the person and the amplitude more than brain state?

Written 2026-11-20 before any run. Subjects: the first 200 is_heldout subjects of the shard index (sorted by ID), so
none was seen in pretraining. Embeddings: the Pilot A anchor windows, plus one window ending at every 30 s epoch
boundary for the stage probe. Encoders: pretrained 10M, random-init (seeds 0-2), and band power (no network).
Inputs: as stored (causal per-night z-score) and per-window renormalised.

Probes (multinomial logistic regression, C fixed at 1.0, standardised features):
- Subject ID: train on windows from the first 40% of each night, test on the last 40%, 20% gap. Top-1 accuracy;
  chance 1/200.
- Amplitude: raw C4 RMS (microvolts) of the 64 s window from the EDF, binned into quintiles over all windows; train
  on 160 subjects, test on 40 others; accuracy, chance 0.2.
- Sleep stage (W, N1, N2, N3+N4, REM) of the epoch the window ends in; same subject split; balanced accuracy.

Primary: subject-ID accuracy, pretrained minus random-init (mean of 3 seeds), as-is input; subject bootstrap over
the 200 subjects. Fingerprint finding if all three hold: that interval lies above 0; pretrained subject-ID accuracy is
at least 2 times random-init; pretrained stage balanced accuracy is at most random-init + 0.02. Amplitude finding: the
amplitude probe is above 0.4 on as-is input and falls by half with per-window renormalisation. Everything else is
descriptive; Holm over the 6 probe x input cells for "above chance".

Kill rule: none, it is a measurement. If the fingerprint finding holds, Pilot A's fine-tune gain must be read against
its shift control before any scaling decision.
