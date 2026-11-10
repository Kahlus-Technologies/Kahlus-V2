# Conformal alarm budget: dev-slice result (2026-11-05)

Status: **dev slice, method works, alarms are not useful yet**. Table regenerated from a fresh clone after the beat quality mask. Refs #89. Code: `src/kahlus/conformal.py`, `scripts/run_conformal_alarm.py`. Numbers: `artifacts/conformal_alarm_dev25/report.json`.

## Setup
- Forecaster: prior+cardiac at h=30 s, causal-normalized (from #83). Anchors every 5 s.
- Within cohort: subjects split 50/25/25 into train, calibrate, test, averaged over 5 seeds. Across cohorts: train and calibrate on one cohort, test on all of the other.
- Threshold: split conformal on calibration positives, so a new positive is missed with probability at most alpha if positives are exchangeable.

## Results (miss rate, false alarms per hour of sleep)

| | alpha 0.1 | 0.2 | 0.3 | 0.5 |
|---|---|---|---|---|
| MESA within | 0.123, 539 | 0.238, 436 | 0.344, 344 | 0.532, 214 |
| SHHS within | 0.071, 550 | 0.183, 433 | 0.267, 359 | 0.444, 238 |
| MESA to SHHS | 0.127, 491 | 0.209, 418 | 0.332, 318 | 0.558, 169 |
| SHHS to MESA | 0.084, 561 | 0.157, 490 | 0.219, 435 | 0.404, 293 |

- **Overall coverage roughly holds.** The realized miss rate stays within about 4 points of alpha (worst: MESA 0.123 at 0.1), even across cohorts. MESA runs slightly over (0.123 at alpha 0.1), which is the expected cost of positives within a night not being independent and only 6 calibration subjects.
- **Per-subject coverage does not.** At alpha 0.2 the per-subject miss rate spans 0.07 to 0.36 (p10 to p90, pooled over 5 seeds) in MESA. A guarantee on average is not a guarantee for the person wearing the device. Subject-level (Mondrian or hierarchical) calibration is the fix, which needs more subjects than 25.
- **The alarms are too frequent to use.** At 5 s anchors there are 720 anchors per hour, so 400 to 550 false alarms per hour means alarming on 60 to 85% of anchors. The prior-only model gives almost the same table (MESA alpha 0.1: 0.120 miss, 530 FA/h; `artifacts/conformal_alarm_dev25_prior/report.json`), consistent with #83: the cardiac block adds nothing here.

## What this changes
- The conformal machinery is correct (unit test confirms the bound on exchangeable data) and cheap. It is ready to wrap any better forecaster.
- With AUROC around 0.63 at h=30 s, no threshold gives both few misses and few false alarms. The paper angle is the coverage gap, not the alarms: marginal conformal coverage looks fine while per-subject coverage is poor and nobody reports it for sleep events.
- Next: Mondrian calibration by subject group (AHI, age) on full cohorts, and alarm rate per hour as a fixed budget instead of a miss rate (conformal risk control).
