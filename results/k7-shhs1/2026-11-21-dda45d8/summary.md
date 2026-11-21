# K7 on SHHS-1 (confirmatory cell): do trigger times come before scored arousals?

SHHS-1 without dev nights 200001 to 200025: 5768 nights, 4,720,440 matched clicks. Preregistration `docs/results/k7_k1_s1_preregistration.md` section 1, SHA-256 `9391ffad657b9395eb6ba1885291bcbe98b3755df3880de202e7a2fda6c7a94a`. One Engram CPU job (32 CPUs, no GPU), single start, reading the shared NSRR mirror.

**Result (primary cell).** Trigger times come before scored arousals: a scored spontaneous arousal starts within 3 s after 1.78% of would-be clicks and after 0.35% of matched control moments. Relative risk **5.01 [4.83, 5.18]** (2,000 subject-level bootstrap resamples). This is a replay measurement, not a causal result about clicks (no sound is played), and it names no mechanism. In SHHS "spontaneous" means "not respiratory" (no leg channel). Slow waves and K-complexes as part of micro-arousals are known (Halász 2004, 2005).

**Gate and finding rule.** Rotated-trigger gate: PASS, rotated relative risk 0.995 [0.962, 1.026]. Clicks dropped by matching: 4.18% (limit 5%). Finding rule parts: interval excludes 1: True; relative risk at least 1.10 or at most 0.91: True; gate passes: True; odd and even subject-ID halves agree in direction with intervals excluding 1: True.

**Heart rate (secondary, no scorer).** Mean rate (t+2, t+10] s minus (t-10, t] s, click minus control: 0.56 bpm [0.53, 0.59]. A slow wave has its own heart response, so this checks direction only.

**Secondary, not a claim.** The trigger-setting grid, percentile threshold, first click of each run, controls without the no-click rule, windows 1/5/10 s, any arousal, and control seeds 1 to 4 are in the tables below and in `result.json`. The "second crossing within 2.5 s" variant of the preregistered grid was not run (6 of 12 settings). Kill rule (every setting within 0.9 to 1.1 on the tuning subjects) fired: False. Tuning and other-subject rows are in `result.json`; no setting was selected from them.

**Provenance.** Commit `dda45d8e842112d73ff48bf4c77acadb15b92f3d`, config SHA-256 `06d67b1e4235f813fd830e797e343539b3bb3aa026cdb7f64deb09b012632b3c`, data manifest SHA-256 `{"shhs": "58e72a5590b5000d577b4a4ed67521b12cef1a2031adf4f5f99f7e67a4fffd67"}`, numpy 2.5.5, scipy 1.18.1, starts 1, GPU-hours 0.

**Reproduce.** `K7_COMMIT=$(git rev-parse HEAD) sky launch -c kahlus-k7 --down -i 1 -y --env K7_COMMIT scripts/engram/k7_shhs1.yaml` (needs the NSRR mirror at `/shared/aayu/nsrr`).

### SHHS: 5768 nights

Rotated-trigger gate: PASS. Clicks dropped by matching: 4.18% (ok).

| setting | clicks/night (median) | click rate | control rate | relative risk [95% CI] | Holm p |
|---|---|---|---|---|---|
| primary (primary) | 716 | 0.0178 | 0.0035 | 5.011 [4.830, 5.179] |  |
| t40_n3 | 306 | 0.0081 | 0.0016 | 4.961 [4.453, 5.403] | 0.002 |
| t60_n23 | 163 | 0.0208 | 0.0053 | 3.894 [3.751, 4.037] | 0.002 |
| t60_n3 | 61 | 0.0111 | 0.0026 | 4.230 [4.008, 4.453] | 0.002 |
| t80_n23 | 39 | 0.0195 | 0.0072 | 2.730 [2.370, 3.082] | 0.002 |
| t80_n3 | 10 | 0.0130 | 0.0042 | 3.122 [2.230, 3.985] | 0.002 |
| pct | 3024 | 0.0123 | 0.0029 | 4.276 [4.054, 4.518] |  |
| first | 511 | 0.0216 | 0.0040 | 5.354 [5.223, 5.487] |  |
| anyctrl | 716 | 0.0174 | 0.0078 | 2.241 [2.206, 2.278] |  |
| rot | 255 | 0.0062 | 0.0062 | 0.995 [0.962, 1.026] |  |
| seed1 | 716 | 0.0178 | 0.0035 | 5.010 [4.824, 5.178] |  |
| seed2 | 716 | 0.0178 | 0.0036 | 4.965 [4.787, 5.124] |  |
| seed3 | 716 | 0.0178 | 0.0036 | 4.990 [4.805, 5.156] |  |
| seed4 | 716 | 0.0178 | 0.0036 | 4.974 [4.790, 5.138] |  |

Primary setting, other windows and the heart-rate outcome (secondary):

| outcome | click | control | estimate [95% CI] |
|---|---|---|---|
| sp3 (relative risk) | 0.0178 | 0.0035 | 5.011 [4.830, 5.179] |
| hr (mean bpm difference) | 0.4818 | -0.0747 | 0.556 [0.530, 0.586] |
| sp1 (relative risk) | 0.0081 | 0.0010 | 7.717 [7.484, 7.963] |
| sp5 (relative risk) | 0.0239 | 0.0074 | 3.220 [3.138, 3.301] |
| sp10 (relative risk) | 0.0356 | 0.0201 | 1.770 [1.740, 1.801] |
| any3 (relative risk) | 0.0274 | 0.0056 | 4.875 [4.734, 5.016] |
