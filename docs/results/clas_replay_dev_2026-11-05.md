# Offline CLAS replay with an arousal-risk veto: dev-slice result (2026-11-05)

Status: **dev slice, positive for a context veto, mixed for the heart**. Table regenerated after the review fixes (causal beats, trailing quality mask). Refs #96. Code: `src/neurotwin/forecastability/clas_replay.py`, `scripts/run_clas_replay.py`. Numbers: `artifacts/clas_replay_dev25/report.json`.

## Setup
- Trigger: causal 0.5-4 Hz filter on C4 EEG, click 0.5 s after a trough below -40 uV, 2.5 s refractory, only where the last completed epoch was N2/N3. No click is ever played; this is a replay.
- Bad click: a spontaneous arousal starts within 3 s after it.
- Veto: drop the top 10/20/30% of clicks by predicted arousal risk. Risk models trained on the other cohort (MESA nights scored by an SHHS-trained model and the reverse). Compared against random drops at the same rate (200 draws) and against a half-night shift of the beat features.

## Results
About 21k clicks per cohort, of which 1.6% (MESA) and 1.8% (SHHS) are bad.

Bad clicks removed, and enrichment over random:

| cohort, drop | random | prior | beats | prior+beats | prior+beats_shift |
|---|---|---|---|---|---|
| MESA 10% | 33 | 65 (x1.96) | 39 (x1.17) | 53 (x1.59) | 64 (x1.93) |
| MESA 20% | 66 | 129 (x1.96) | 72 (x1.09) | 120 (x1.82) | 131 (x1.99) |
| MESA 30% | 98 | 168 (x1.71) | 121 (x1.23) | 186 (x1.89) | 173 (x1.76) |
| SHHS 10% | 37 | 66 (x1.77) | 63 (x1.69) | 81 (x2.17) | 68 (x1.82) |
| SHHS 20% | 73 | 134 (x1.82) | 104 (x1.42) | 160 (x2.18) | 134 (x1.82) |
| SHHS 30% | 110 | 185 (x1.68) | 142 (x1.29) | 222 (x2.01) | 205 (x1.86) |

Random p95 is roughly 1.2x the random mean, so every prior-based veto clears chance, in both transfer directions.

## Read
- **A veto works, mostly from context.** The prior-context model (time since last arousal, time since sleep onset, last completed stage) nearly doubles the rate of avoided pre-arousal clicks in both directions. Adding beat features beats the shift control on SHHS (x2.0 to 2.2 vs x1.8 to 1.9) but not on MESA (mixed, within noise). Consistent with the small beat-level signal in #83 Result 6; not established.
- **Beats alone do beat random (x1.3 to 1.7)**, but beats carry the same person and drift information the shift control exposed, so this is not evidence of a precursor.
- **Cost:** dropping 20% of clicks avoids about 35% of bad clicks (MESA 114/330, SHHS 156/368). Whether that trade is worth it depends on how much a pre-arousal click actually hurts, which only a live study can answer.
- Only 1.6 to 1.8% of clicks are bad to begin with, so the absolute gain is small. The stronger version of this study measures clicks within 3 s before *any* arousal and weights by stimulation dose.

## Next
- Full cohorts, per-night confidence intervals, and a window sweep (1, 3, 5 s).
- An EEG-based risk feature (pre-arousal alpha/beta rise) is the obvious next veto input, since the trigger already reads EEG.
