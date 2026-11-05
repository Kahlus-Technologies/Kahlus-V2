# Offline CLAS replay with an arousal-risk veto: dev-slice result (2026-11-05)

Status: **dev slice, positive for a context veto, null for the heart**. Refs #96. Code: `src/neurotwin/forecastability/clas_replay.py`, `scripts/run_clas_replay.py`. Numbers: `artifacts/clas_replay_dev25/report.json`.

## Setup
- Trigger: causal 0.5-4 Hz filter on C4 EEG, click 0.5 s after a trough below -40 uV, 2.5 s refractory, only where the last completed epoch was N2/N3. No click is ever played; this is a replay.
- Bad click: a spontaneous arousal starts within 3 s after it.
- Veto: drop the top 10/20/30% of clicks by predicted arousal risk. Risk models trained on the other cohort (MESA nights scored by an SHHS-trained model and the reverse). Compared against random drops at the same rate (200 draws) and against a half-night shift of the beat features.

## Results
About 21k clicks per cohort, of which 1.6% (MESA) and 1.8% (SHHS) are bad.

Bad clicks removed, and enrichment over random:

| cohort, drop | random | prior | beats | prior+beats | prior+beats_shift |
|---|---|---|---|---|---|
| MESA 10% | 33 | 54 (x1.62) | 27 (x0.81) | 40 (x1.20) | 60 (x1.80) |
| MESA 20% | 66 | 112 (x1.70) | 93 (x1.41) | 114 (x1.73) | 114 (x1.73) |
| MESA 30% | 98 | 181 (x1.84) | 129 (x1.31) | 177 (x1.80) | 178 (x1.81) |
| SHHS 10% | 37 | 75 (x2.01) | 64 (x1.71) | 74 (x1.98) | 82 (x2.19) |
| SHHS 20% | 74 | 141 (x1.92) | 115 (x1.56) | 156 (x2.12) | 152 (x2.07) |
| SHHS 30% | 110 | 199 (x1.80) | 158 (x1.43) | 218 (x1.97) | 210 (x1.90) |

Random p95 is 44 to 126, so every prior-based veto clears chance, in both transfer directions.

## Read
- **A veto works, but the heart is not why.** The prior-context model (time since last arousal, time since sleep onset, last completed stage) nearly doubles the rate of avoided pre-arousal clicks. Adding beat features does not help, and the shifted beats do as well as the aligned ones. This matches #83.
- **Beats alone do beat random (x1.3 to 1.7)**, but beats carry the same person and drift information the shift control exposed, so this is not evidence of a precursor.
- **Cost:** dropping 20% of clicks avoids about 35% of bad clicks (MESA 114/330, SHHS 156/368). Whether that trade is worth it depends on how much a pre-arousal click actually hurts, which only a live study can answer.
- Only 1.6 to 1.8% of clicks are bad to begin with, so the absolute gain is small. The stronger version of this study measures clicks within 3 s before *any* arousal and weights by stimulation dose.

## Next
- Full cohorts, per-night confidence intervals, and a window sweep (1, 3, 5 s).
- An EEG-based risk feature (pre-arousal alpha/beta rise) is the obvious next veto input, since the trigger already reads EEG.
