# Conformal alarm budget: dev-slice result (2026-11-05)

Status: **dev slice, method works, alarms are not useful yet**. Refs #89. Code: `src/neurotwin/forecastability/conformal.py`, `scripts/run_conformal_alarm.py`. Numbers: `artifacts/conformal_alarm_dev25/report.json`.

## Setup
- Forecaster: prior+cardiac at h=30 s, causal-normalized (from #83). Anchors every 5 s.
- Within cohort: subjects split 50/25/25 into train, calibrate, test, averaged over 5 seeds. Across cohorts: train and calibrate on one cohort, test on all of the other.
- Threshold: split conformal on calibration positives, so a new positive is missed with probability at most alpha if positives are exchangeable.

## Results (miss rate, false alarms per hour of sleep)

| | alpha 0.1 | 0.2 | 0.3 | 0.5 |
|---|---|---|---|---|
| MESA within | 0.129, 519 | 0.233, 425 | 0.344, 329 | 0.550, 181 |
| SHHS within | 0.067, 540 | 0.150, 446 | 0.244, 364 | 0.456, 225 |
| MESA to SHHS | 0.091, 496 | 0.186, 409 | 0.311, 302 | 0.520, 179 |
| SHHS to MESA | 0.077, 547 | 0.169, 453 | 0.240, 379 | 0.412, 245 |

- **Overall coverage roughly holds.** The realized miss rate stays within about 3 to 5 points of alpha, even across cohorts. MESA runs slightly over (0.129 at alpha 0.1), which is the expected cost of positives within a night not being independent and only 6 calibration subjects.
- **Per-subject coverage does not.** At alpha 0.2 the per-subject miss rate spans 0.19 to 0.38 (p10 to p90) in MESA. A guarantee on average is not a guarantee for the person wearing the device. Subject-level (Mondrian or hierarchical) calibration is the fix, which needs more subjects than 25.
- **The alarms are too frequent to use.** At 5 s anchors there are 720 anchors per hour, so 400 to 550 false alarms per hour means alarming on 60 to 85% of anchors. The prior-only model gives almost the same table (e.g. MESA alpha 0.1: 0.124 miss, 513 FA/h), consistent with #83: the cardiac block adds nothing here.

## What this changes
- The conformal machinery is correct (unit test confirms the bound on exchangeable data) and cheap. It is ready to wrap any better forecaster.
- With AUROC around 0.63 at h=30 s, no threshold gives both few misses and few false alarms. The paper angle is the coverage gap, not the alarms: marginal conformal coverage looks fine while per-subject coverage is poor and nobody reports it for sleep events.
- Next: Mondrian calibration by subject group (AHI, age) on full cohorts, and alarm rate per hour as a fixed budget instead of a miss rate (conformal risk control).
