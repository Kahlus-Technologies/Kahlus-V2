# Heart-only arousal forecasting: dev-slice result (2026-11-05)

Status: **dev slice, negative after controls**. 25 MESA + 25 SHHS1 nights. Refs #83, #84. Bottom line (Result 5): once a shift control removes person-identity effects, time-aligned cardiac features add essentially nothing for 5-120 s arousal forecasting. Results 1-4 are kept as the path to that conclusion.

## Setup
- Anchor every 5 s during sleep with 60 s of quiet before it (no arousal, respiratory event, limb movement, or wake). Inputs end at the anchor. The stage score of the epoch containing the anchor is never used.
- Target: the first arousal in (t, t+h] is spontaneous. Windows where the first arousal is respiratory or limb-related are excluded.
- Models (one fixed gradient-boosting config): prior context (time since last arousal, time since sleep onset, last completed stage), plus cardiac RR features, EEG band power, both, or PPG (MESA only).
- Splits: subject-grouped 5-fold CV inside each cohort, plus MESA to SHHS and SHHS to MESA transfer.
- Code: `src/neurotwin/forecastability/heart_arousal.py`, `scripts/run_heart_arousal_benchmark.py`. Raw numbers: `artifacts/heart_arousal_dev25/report.json`.

## Result 1: cardiac features add nothing over prior context
Residual bits of prior+cardiac over prior (subject-bootstrap 95% CI):

| horizon | MESA CV | SHHS CV | MESA to SHHS | SHHS to MESA |
|---|---|---|---|---|
| 5 s | +0.0001 [-0.0010, +0.0015] | +0.0006 [-0.0001, +0.0012] | +0.0003 [-0.0017, +0.0022] | -0.0013 [-0.0022, -0.0004] |
| 15 s | +0.0004 | +0.0008 | -0.0021 | -0.0040 |
| 30 s | -0.0076 | -0.0001 | -0.0054 | -0.0166 |
| 60 s | -0.0140 | -0.0116 | -0.0070 | -0.0161 |
| 120 s | -0.0327 | -0.0037 | -0.0335 | -0.0130 |

The prior-only model reaches AUROC 0.67 to 0.68 at 5 s and 0.59 to 0.63 at 120 s. Adding cardiac features moves AUROC by at most about 0.02 and never clears zero bits with a CI that excludes zero. At long horizons extra features overfit (negative bits).

## Result 2: why. HR moves only about 1 s before onset
Event-locked HR around clean spontaneous arousals (`scripts/event_locked_hr.py`, `artifacts/event_locked_hr/report.json`):

| | n | -10 s | -5 s | -3 s | -1 s | 0 s | +2 s | +5 s |
|---|---|---|---|---|---|---|---|---|
| MESA | 725 | -0.36 | +0.15 | +0.37 | +0.48 | +1.33 | +3.51 | +5.80 bpm |
| SHHS | 679 | -0.11 | -0.11 | +0.32 | +1.06 | +1.80 | +3.77 | +5.67 bpm |

The 95% CI first excludes zero 1 s before scored onset, which is inside the interpolation look-ahead of about 1 beat. The precursor that Sforza 2000 and Bonnet and Arand describe (rise before cortical onset) is real but only a few beats long. A 5 s horizon is already past most of it.

## Result 3: absolute EEG power does not transfer
prior+EEG drops by 0.03 to 0.15 bits when trained on MESA and tested on SHHS (h=30 to 120 s), while it helps slightly within SHHS. Amplifier gain and montage differ between cohorts, so `eeg_log_power` is a cohort fingerprint. Use relative power only, or normalize per night, before any transfer claim.

## Result 4: per-night normalization changes the answer
Same run with cardiac/EEG/PPG features z-scored within each night (`--normalize night`, `artifacts/heart_arousal_dev25_nightz/report.json`). Cardiac bits over prior:

| horizon | MESA CV | SHHS CV | MESA to SHHS | SHHS to MESA |
|---|---|---|---|---|
| 5 s | +0.0001 | +0.0004 | **+0.0022 [+0.0011, +0.0035]** | +0.0002 |
| 15 s | +0.0035 [-0.0001, +0.0096] | +0.0036 | +0.0016 | +0.0025 |
| 30 s | +0.0027 | **+0.0103 [+0.0041, +0.0168]** | +0.0020 | +0.0072 |
| 60 s | -0.0066 | +0.0084 | +0.0038 | +0.0049 |
| 120 s | +0.0095 | +0.0123 | +0.0020 | **+0.0225 [+0.0039, +0.0442]** |

EEG transfer also stops collapsing (MESA to SHHS at 120 s goes from -0.153 to +0.0035 bits). So most of Result 1 was a units problem: absolute HR and power differ by person and amplifier, and a raw-feature model learns the person. What the heart carries is *relative change within a night*. 19 of 20 cardiac cells are now positive, but only 3 have a CI that excludes zero, and 20 cells at 25+25 nights invite false positives. The full-cohort claim-mode run decides it.

Caveat: the z-score uses the whole night's mean and SD, a mild within-night look-ahead. It is the same for every model compared. A causal running normalizer is needed before any deployment claim.

## Result 5: a shift control explains the Result 4 positives
Two more runs (`artifacts/heart_arousal_dev25_causal/report.json`):
- **Causal normalizer:** z-score each anchor against earlier anchors in the same night only (5 min warmup). This removes the whole-night look-ahead in Result 4. Cardiac bits over prior grew: 20/20 cells non-negative, 10 with CI > 0.
- **Shift control:** the same cardiac features rotated by half a night inside each night (`prior+cardiac_shift`). They keep each person's distribution, drift and dropout pattern but lose alignment with the labels. This control *also* gained bits over prior, so the Result 4 gains were mostly the model recognizing the person (and their arousal rate), not reading a precursor.

Headline metric, aligned cardiac bits = prior+cardiac vs prior+cardiac_shift:

| horizon | MESA CV | SHHS CV | MESA to SHHS | SHHS to MESA |
|---|---|---|---|---|
| 5 s | +0.0005 [-0.0005, +0.0016] | +0.0005 [-0.0001, +0.0012] | -0.0003 | +0.0007 |
| 15 s | +0.0017 | +0.0006 | -0.0010 | -0.0036 [-0.0073, -0.0006] |
| 30 s | **+0.0051 [+0.0010, +0.0096]** | +0.0030 | -0.0016 | -0.0020 |
| 60 s | +0.0053 | +0.0039 | -0.0009 | -0.0047 |
| 120 s | **+0.0168 [+0.0004, +0.0407]** | +0.0026 | +0.0036 | -0.0006 |

2 of 20 cells clear zero, both within MESA, and none transfer. With 20 cells, about 1 false positive is expected at this CI width. Call it null on the dev slice.

## What this changes
- Every future comparison must include the shift control. Without it, any per-person feature looks predictive.
- Normalization still matters for EEG transfer (Result 3 vs Result 4), but it does not create cardiac signal.
- Minutes-ahead heart forecasting is weak at best. The sharper question is **beat-level, 0 to 5 s lead**: does the pre-onset HR rise give a usable 1 to 3 s warning? That is still useful for closed-loop stimulation (#96), where you only need to suppress the next click.
- The prior-context baseline is strong and must be the reference in every comparison. Arousals cluster, and a model can lean on that.
- Next: (1) sub-5 s horizons with beat-resolved features (last 3 to 5 RR intervals, no interpolation), (2) full cohorts on the cluster with claim-mode bootstrap. Scorer onset jitter (kappa about 0.56 to 0.60 for arousals) is a ceiling on any second-scale result.
- SHHS 200002 and MESA 0010 show median HR over 110 bpm late in the night, but those hours are scored wake (MESA PPG is flat there too, so the sensors were off). Anchors never fall in wake. `good_beat_mask` now drops such 5-min blocks anyway; it touches 0.02-0.03% of sleep beats.
