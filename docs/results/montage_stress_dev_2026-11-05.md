# Low-cost montage stress test: dev-slice result (2026-11-05)

Status: **dev slice, method works, baseline detector is weak**. Table regenerated after the arousal-label end fix. Refs #97. Code: `src/kahlus/montage_degrade.py`, `scripts/run_montage_stress.py`. Numbers: `artifacts/montage_stress_dev25/report.json`.

## Setup
- Detector: one C4 channel, per-second features from the 3 s of EEG ending at t (relative band power, log power, plus a causal within-night z-score), one fixed gradient-boosting model. Trained on clean PSG from one cohort, tested on the other.
- Degradations of the *test* EEG: white noise (5, 10, 20 uV RMS), and two dry-headband profiles combining noise, slow drift and electrode pops (1 or 4 per minute).
- Event scoring: detected onsets matched one-to-one to scored arousal onsets during sleep within 3 s. Threshold tuned for event F1 on the training cohort, then frozen.

## Results

| level | MESA to SHHS: F1 / precision / recall / false det. per h | SHHS to MESA |
|---|---|---|
| clean | 0.251 / 0.193 / 0.357 / 28 | 0.204 / 0.144 / 0.351 / 43 |
| noise 5 uV | 0.208 / 0.136 / 0.446 / 53 | 0.195 / 0.138 / 0.333 / 42 |
| noise 10 uV | 0.115 / 0.064 / 0.548 / 149 | 0.153 / 0.099 / 0.335 / 62 |
| noise 20 uV | 0.081 / 0.044 / 0.517 / 209 | 0.110 / 0.064 / 0.406 / 122 |
| dry headband | 0.104 / 0.058 / 0.487 / 147 | 0.129 / 0.086 / 0.257 / 56 |
| bad dry headband | 0.068 / 0.037 / 0.402 / 193 | 0.086 / 0.052 / 0.251 / 93 |

Median timing error of matched events is 1.5 to 2.1 s throughout.

## Read
- **Degradation mostly costs precision, not recall.** Noise looks like high-frequency power, which is what an arousal looks like on EEG, so the detector fires more often. False detections rise 2 to 7x; F1 roughly halves at a realistic dry-headband level.
- **The clean baseline is weak (F1 0.20 to 0.24 across cohorts).** Published multichannel detectors reach AUPRC 0.6 to 0.85. A single 3 s window, one channel and hand-made features are the ceiling here, so absolute numbers say little about real headbands. The useful output is the *shape* of the curve and the event-level reporting standard (F1, timing error, false detections per hour), which the gap scan found nobody reports for consumer devices.
- **Noise robustness depends on training data.** SHHS-trained (noisier, 125 Hz source) held up better to 5 uV noise than MESA-trained. Training on degraded copies is the cheap fix to try first.

## Next
- Train with degradation augmentation and rerun the curve.
- Stronger single-channel detector (longer context, a small CNN) so the baseline is credible.
- Real device check: the Cerelog/ADS1299 rig from #92 recorded alongside PSG. That needs IRB before any human pilot.
