# NSRR arousal census (2026-11-04)

Script: `scripts/nsrr_arousal_census.py` (with self-check). Input: every NSRR annotation XML, MESA (2,056) and SHHS1 (5,792; one XML failed to download).
Subtype comes from timing. Respiratory = arousal starts within 5 s of an apnea/hypopnea. Limb = arousal starts within 0.5 s of a limb movement. Clean spontaneous = spontaneous, in sleep, with no arousal, wake, respiratory or limb event in the previous 60 s.

| dataset | subjects | arousals | respiratory | limb | spontaneous | clean spontaneous | median clean / subject | subjects with >=10 clean |
|---|---|---|---|---|---|---|---|---|
| MESA | 2,056 | 323,452 | 82,283 | 22,687 | 218,482 (67.5%) | 57,767 | 27 | 1,819 |
| SHHS1 | 5,792 | 744,043 | 300,591 | 0* | 443,452 (59.6%) | 144,533 | 23 | 4,910 |

*SHHS has no leg channel.

Takeaways:
- The pre-registered primary target for #83 (clean spontaneous arousals) has enough events for subject-level inference in both cohorts.
- For the SHHS transfer (#84), use ECG only. SHHS has no PPG, so any feature that depends on Pleth cannot transfer. Channel map: `configs/data/nsrr_common_montage.yaml`.
- Sensitivity sweep of the 5 s and 60 s windows still to do.
