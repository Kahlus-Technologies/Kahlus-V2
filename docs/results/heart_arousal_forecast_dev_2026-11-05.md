# Heart-only arousal forecasting: dev-slice result (2026-11-05)

Status: **dev slice, not a claim**. 25 MESA + 25 SHHS1 nights. Refs #83, #84. Raw features: no cardiac signal. Per-night z-scored features: a small positive that sometimes clears zero (Result 4).

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
Same run with cardiac/EEG/PPG features z-scored within each night (`--per-night-z`, `artifacts/heart_arousal_dev25_nightz/report.json`). Cardiac bits over prior:

| horizon | MESA CV | SHHS CV | MESA to SHHS | SHHS to MESA |
|---|---|---|---|---|
| 5 s | +0.0001 | +0.0004 | **+0.0022 [+0.0011, +0.0035]** | +0.0002 |
| 15 s | +0.0035 [-0.0001, +0.0096] | +0.0036 | +0.0016 | +0.0025 |
| 30 s | +0.0027 | **+0.0103 [+0.0041, +0.0168]** | +0.0020 | +0.0072 |
| 60 s | -0.0066 | +0.0084 | +0.0038 | +0.0049 |
| 120 s | +0.0095 | +0.0123 | +0.0020 | **+0.0225 [+0.0039, +0.0442]** |

EEG transfer also stops collapsing (MESA to SHHS at 120 s goes from -0.153 to +0.0035 bits). So most of Result 1 was a units problem: absolute HR and power differ by person and amplifier, and a raw-feature model learns the person. What the heart carries is *relative change within a night*. 19 of 20 cardiac cells are now positive, but only 3 have a CI that excludes zero, and 20 cells at 25+25 nights invite false positives. The full-cohort claim-mode run decides it.

Caveat: the z-score uses the whole night's mean and SD, a mild within-night look-ahead. It is the same for every model compared. A causal running normalizer is needed before any deployment claim.

## What this changes
- Per-night (or running) normalization is mandatory for every signal block. Raw-unit results in Result 1 and Result 3 are superseded.
- Minutes-ahead heart forecasting is weak at best. The sharper question is **beat-level, 0 to 5 s lead**: does the pre-onset HR rise give a usable 1 to 3 s warning? That is still useful for closed-loop stimulation (#96), where you only need to suppress the next click.
- The prior-context baseline is strong and must be the reference in every comparison. Arousals cluster, and a model can lean on that.
- Next: (1) sub-5 s horizons with beat-resolved features (last 3 to 5 RR intervals, no interpolation), (2) a causal running normalizer, (3) full cohorts on the cluster with claim-mode bootstrap. Scorer onset jitter (kappa about 0.56 to 0.60 for arousals) is a ceiling on any second-scale result.
- One SHHS night (200002) has QRS double-counting from hour 4 on (median 156 bpm). Needs a per-night ECG quality gate.
