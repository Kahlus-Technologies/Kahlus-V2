# K7 on sealed MESA (the one final look): do trigger times come before scored arousals?

MESA, never-seen nights only (2,056 nights on the mirror minus the 25 exposed dev nights, so 2031); the seal read no exposed night. This is the single final look for `run_k7` (ledger `docs/results/mesa_final_looks/run_k7.json`, commit `3a895b0`); a dev-mode dry run on the 25 exposed nights ran first and did not touch the seal. No model is fitted and no setting was chosen on MESA: the primary setting is the one fixed in the preregistration. 2031 nights, 1,602,446 matched clicks. Preregistration `docs/results/k7_k1_s1_preregistration.md` section 1, SHA-256 `9391ffad657b9395eb6ba1885291bcbe98b3755df3880de202e7a2fda6c7a94a`. One Engram CPU job (32 CPUs, no GPU), single start, reading the shared NSRR mirror.

**Result (primary cell).** Trigger times come before scored arousals: a scored spontaneous arousal starts within 3 s after 1.65% of would-be clicks and after 0.44% of matched control moments. Relative risk **3.79 [3.56, 4.02]** (2,000 subject-level bootstrap resamples). This is a replay measurement, not a causal result about clicks (no sound is played), and it names no mechanism. In MESA "spontaneous" means not respiratory and not limb-related. Slow waves and K-complexes as part of micro-arousals are known (Halász 2004, 2005).

**Gate and finding rule.** Rotated-trigger gate: PASS, rotated relative risk 1.004 [0.945, 1.070]. Clicks dropped by matching: 3.51% (limit 5%). Finding rule parts: interval excludes 1: True; relative risk at least 1.10 or at most 0.91: True; gate passes: True; odd and even subject-ID halves agree in direction with intervals excluding 1: True.

**Heart rate (secondary, no scorer).** Mean rate (t+2, t+10] s minus (t-10, t] s, click minus control: 0.66 bpm [0.63, 0.70]. A slow wave has its own heart response, so this checks direction only.

**Secondary, not a claim.** The trigger-setting grid, percentile threshold, first click of each run, controls without the no-click rule, windows 1/5/10 s, any arousal, and control seeds 1 to 4 are in the tables below and in `result.json`. The "second crossing within 2.5 s" variant of the preregistered grid was not run (6 of 12 settings). Kill rule (every setting within 0.9 to 1.1 on the tuning subjects) fired: False. Tuning and other-subject rows are in `result.json`; no setting was selected from them.

**Provenance.** Commit `3a895b0689f2565c1cc146d372adcc5b22d7e214` (clean clone, seal check passed), config SHA-256 `06d67b1e4235f813fd830e797e343539b3bb3aa026cdb7f64deb09b012632b3c`, data manifest SHA-256 `{"mesa": "59c1c1de4614595d514c40b2aadc41f057c7bd283911e1a060f40552443c7faf"}`, numpy 2.5.5, scipy 1.18.1, starts 1, GPU-hours 0.

**Reproduce.** `from a clean clone: `K7_COMMIT=$(git rev-parse HEAD) sky launch -c k7-mesa --down -i 1 -y --env K7_COMMIT scripts/engram/k7_mesa.yaml` (single use: it cannot be rerun).

### MESA: 2031 nights

Rotated-trigger gate: PASS. Clicks dropped by matching: 3.51% (ok).

| setting | clicks/night (median) | click rate | control rate | relative risk [95% CI] | Holm p |
|---|---|---|---|---|---|
| primary (primary) | 649 | 0.0165 | 0.0044 | 3.786 [3.560, 4.020] |  |
| t40_n3 | 150 | 0.0071 | 0.0019 | 3.728 [3.200, 4.311] | 0.002 |
| t60_n23 | 155 | 0.0170 | 0.0072 | 2.381 [2.144, 2.627] | 0.002 |
| t60_n3 | 30 | 0.0093 | 0.0028 | 3.287 [2.854, 3.737] | 0.002 |
| t80_n23 | 56 | 0.0140 | 0.0092 | 1.518 [1.345, 1.689] | 0.002 |
| t80_n3 | 7 | 0.0107 | 0.0041 | 2.572 [2.021, 3.236] | 0.002 |
| pct | 2387 | 0.0139 | 0.0029 | 4.745 [4.382, 5.149] |  |
| first | 450 | 0.0204 | 0.0045 | 4.549 [4.363, 4.742] |  |
| anyctrl | 649 | 0.0163 | 0.0086 | 1.900 [1.856, 1.946] |  |
| rot | 126 | 0.0071 | 0.0070 | 1.004 [0.945, 1.070] |  |
| seed1 | 649 | 0.0165 | 0.0044 | 3.773 [3.541, 4.006] |  |
| seed2 | 649 | 0.0165 | 0.0044 | 3.777 [3.547, 4.010] |  |
| seed3 | 649 | 0.0165 | 0.0043 | 3.816 [3.580, 4.062] |  |
| seed4 | 649 | 0.0165 | 0.0044 | 3.773 [3.546, 4.009] |  |

Primary setting, other windows and the heart-rate outcome (secondary):

| outcome | click | control | estimate [95% CI] |
|---|---|---|---|
| sp3 (relative risk) | 0.0165 | 0.0044 | 3.786 [3.560, 4.020] |
| hr (mean bpm difference) | 0.4890 | -0.1704 | 0.659 [0.626, 0.695] |
| sp1 (relative risk) | 0.0075 | 0.0013 | 5.833 [5.359, 6.324] |
| sp5 (relative risk) | 0.0228 | 0.0091 | 2.511 [2.395, 2.628] |
| sp10 (relative risk) | 0.0353 | 0.0235 | 1.506 [1.453, 1.554] |
| any3 (relative risk) | 0.0238 | 0.0060 | 3.946 [3.756, 4.153] |
