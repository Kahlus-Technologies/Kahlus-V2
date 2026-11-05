# Low-cost montage stress test: dev-slice result (2026-11-05)

Status: **dev slice, method works, baseline detector is weak**. Refs #97. Code: `src/neurotwin/forecastability/montage_degrade.py`, `scripts/run_montage_stress.py`. Numbers: `artifacts/montage_stress_dev25/report.json`.

## Setup
- Detector: one C4 channel, per-second features from the 3 s of EEG ending at t (relative band power, log power, plus a causal within-night z-score), one fixed gradient-boosting model. Trained on clean PSG from one cohort, tested on the other.
- Degradations of the *test* EEG: white noise (5, 10, 20 uV RMS), and two dry-headband profiles combining noise, slow drift and electrode pops (1 or 4 per minute).
- Event scoring: detected onsets matched one-to-one to scored arousal onsets during sleep within 3 s. Threshold tuned for event F1 on the training cohort, then frozen.

## Results

| level | MESA to SHHS: F1 / precision / recall / false det. per h | SHHS to MESA |
|---|---|---|
| clean | 0.244 / 0.188 / 0.350 / 28 | 0.200 / 0.139 / 0.356 / 45 |
| noise 5 uV | 0.201 / 0.130 / 0.438 / 54 | 0.196 / 0.138 / 0.339 / 44 |
| noise 10 uV | 0.111 / 0.062 / 0.544 / 153 | 0.155 / 0.098 / 0.368 / 69 |
| noise 20 uV | 0.079 / 0.043 / 0.518 / 216 | 0.098 / 0.055 / 0.454 / 161 |
| dry headband | 0.103 / 0.057 / 0.498 / 153 | 0.132 / 0.086 / 0.279 / 61 |
| bad dry headband | 0.069 / 0.038 / 0.403 / 189 | 0.083 / 0.049 / 0.290 / 116 |

Median timing error of matched events is 1.5 to 2.1 s throughout.

## Read
- **Degradation mostly costs precision, not recall.** Noise looks like high-frequency power, which is what an arousal looks like on EEG, so the detector fires more often. False detections rise 2 to 7x; F1 roughly halves at a realistic dry-headband level.
- **The clean baseline is weak (F1 0.20 to 0.24 across cohorts).** Published multichannel detectors reach AUPRC 0.6 to 0.85. A single 3 s window, one channel and hand-made features are the ceiling here, so absolute numbers say little about real headbands. The useful output is the *shape* of the curve and the event-level reporting standard (F1, timing error, false detections per hour), which the gap scan found nobody reports for consumer devices.
- **Noise robustness depends on training data.** SHHS-trained (noisier, 125 Hz source) held up better to 5 uV noise than MESA-trained. Training on degraded copies is the cheap fix to try first.

## Next
- Train with degradation augmentation and rerun the curve.
- Stronger single-channel detector (longer context, a small CNN) so the baseline is credible.
- Real device check: the Cerelog/ADS1299 rig from #92 recorded alongside PSG. That needs IRB before any human pilot.
