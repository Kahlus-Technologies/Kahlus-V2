# K7 dev-slice pilot: do trigger times come before scored arousals?

Pilot on the 25 MESA + 25 SHHS-1 development nights. The preregistration (`docs/results/k7_k1_s1_preregistration.md`, SHA-256 `9391ffad657b9395eb6ba1885291bcbe98b3755df3880de202e7a2fda6c7a94a`) says a dev-slice table is a pilot, never a claim. The confirmatory cell is SHHS-1 without these nights (see `results/k7-shhs1/` when the Engram run lands).

**Wording.** Trigger times come before scored arousals: in these 50 nights, the causal trigger's would-be click times are followed by a scored spontaneous arousal within 3 s more often than matched moments without a trigger. This is a replay measurement. No sound is played, so it is not a causal result about clicks and it names no mechanism. Slow waves and K-complexes as part of micro-arousals are known (Halász 2004, 2005); this note only measures where the repo's trigger points. In SHHS "spontaneous" means "not respiratory" (no leg channel).

**What passed.** Unit tests pass. The rotated-trigger gate passes in both cohorts (interval of the rotated relative risk contains 1), and matching drops 0.12% (SHHS) and 0.17% (MESA) of clicks (limit 5%). The rotated estimates sit below 1 (SHHS 0.86, MESA 0.70), so a small residual bias is possible; the confirmatory run has the power to show it.

**Not run.** The "second crossing within 2.5 s" trigger variant of the preregistered grid (6 of 12 settings ran: level -40/-60/-80 uV x N2+N3 or N3 only). No setting was chosen from this table. Holm covers the 5 non-primary settings.

**Rules in code.** 5 controls per click from the same night, same stage of the last completed epoch, same time-since-arousal bin, within 600 s (1,800 s if none), at least 3 s from any click and from each other, control draws seed 0 (seeds 1 to 4 in the table). Heart-rate outcome: mean rate in (t+2, t+10] s minus mean in (t-10, t] s, from `causal_beats`. Interval: 2,000 bootstrap resamples over nights (one night per subject), seed 0.

**Provenance.** Commit `f7bf2deffe7ba72fbcf8df41db60ec799f5ca99c` (dirty: False), config SHA-256 `06d67b1e4235f813fd830e797e343539b3bb3aa026cdb7f64deb09b012632b3c`, data manifest SHA-256 `{"mesa": "04c23b5a506473719876dfb8fd09250f62c16eb4ae233c6d0b7d427947fbc987", "shhs": "10fe875cf6b6647915d47e6e4c9904a954d29e741975b47d6a981d8a437ce36a"}`, numpy 2.5.5, scipy 1.18.1, starts 1, GPU-hours 0.

**Reproduce.**
```
scripts/nsrr_fetch_dev_slice_gem.sh            # NSRR gem, token from ~/.nsrr_token, SSL_CERT_FILE=~/.config/nsrr/ca-bundle.pem
PYTHONPATH=src python scripts/run_k7.py --mesa ~/data/nsrr/mesa/polysomnography --shhs ~/data/nsrr/shhs/polysomnography --out /tmp/k7_dev --workers 6
python -m pytest tests/test_clas_replay.py tests/test_rfs.py
```

## Tables (relative risk = click rate over control rate; heart rate is a mean difference in bpm)

### MESA: 25 nights

Rotated-trigger gate: PASS. Clicks dropped by matching: 0.17% (ok).

| setting | clicks/night (median) | click rate | control rate | relative risk [95% CI] | Holm p |
|---|---|---|---|---|---|
| primary (primary) | 686 | 0.0154 | 0.0043 | 3.607 [2.689, 5.012] |  |
| t40_n3 | 166 | 0.0071 | 0.0037 | 1.907 [0.902, 6.058] | 0.188 |
| t60_n23 | 154 | 0.0172 | 0.0063 | 2.725 [1.851, 3.801] | 0.002 |
| t60_n3 | 36 | 0.0075 | 0.0041 | 1.825 [1.227, 2.917] | 0.012 |
| t80_n23 | 56 | 0.0133 | 0.0074 | 1.806 [1.075, 2.570] | 0.087 |
| t80_n3 | 5 | 0.0087 | 0.0051 | 1.724 [0.000, 4.167] | 0.396 |
| pct | 2341 | 0.0130 | 0.0030 | 4.299 [3.230, 6.210] |  |
| first | 448 | 0.0188 | 0.0043 | 4.337 [3.173, 5.718] |  |
| anyctrl | 686 | 0.0155 | 0.0088 | 1.750 [1.484, 2.080] |  |
| rot | 159 | 0.0063 | 0.0090 | 0.698 [0.473, 1.054] |  |
| seed1 | 686 | 0.0154 | 0.0044 | 3.462 [2.561, 4.711] |  |
| seed2 | 686 | 0.0154 | 0.0042 | 3.656 [2.744, 5.038] |  |
| seed3 | 686 | 0.0154 | 0.0045 | 3.448 [2.626, 4.654] |  |
| seed4 | 686 | 0.0154 | 0.0043 | 3.584 [2.602, 5.186] |  |

Primary setting, other windows and the heart-rate outcome (secondary):

| outcome | click | control | estimate [95% CI] |
|---|---|---|---|
| sp3 (relative risk) | 0.0154 | 0.0043 | 3.607 [2.689, 5.012] |
| hr (mean bpm difference) | 0.5166 | -0.2101 | 0.727 [0.573, 0.884] |
| sp1 (relative risk) | 0.0074 | 0.0010 | 7.670 [4.703, 11.977] |
| sp5 (relative risk) | 0.0213 | 0.0092 | 2.316 [1.911, 2.798] |
| sp10 (relative risk) | 0.0333 | 0.0238 | 1.402 [1.243, 1.581] |
| any3 (relative risk) | 0.0211 | 0.0058 | 3.620 [2.904, 4.561] |

### SHHS: 25 nights

Rotated-trigger gate: PASS. Clicks dropped by matching: 0.12% (ok).

| setting | clicks/night (median) | click rate | control rate | relative risk [95% CI] | Holm p |
|---|---|---|---|---|---|
| primary (primary) | 517 | 0.0178 | 0.0033 | 5.335 [3.803, 7.500] |  |
| t40_n3 | 276 | 0.0064 | 0.0008 | 7.629 [3.605, 52.509] | 0.002 |
| t60_n23 | 101 | 0.0218 | 0.0055 | 3.961 [2.081, 6.153] | 0.002 |
| t60_n3 | 64 | 0.0088 | 0.0009 | 9.722 [5.000, 23.643] | 0.002 |
| t80_n23 | 34 | 0.0241 | 0.0075 | 3.210 [1.265, 4.932] | 0.032 |
| t80_n3 | 7 | 0.0099 | 0.0025 | 4.000 [1.643, 7.589] | 0.032 |
| pct | 2730 | 0.0126 | 0.0030 | 4.236 [3.210, 5.810] |  |
| first | 432 | 0.0216 | 0.0043 | 5.046 [3.367, 7.498] |  |
| anyctrl | 517 | 0.0179 | 0.0080 | 2.229 [1.855, 2.755] |  |
| rot | 150 | 0.0068 | 0.0079 | 0.855 [0.635, 1.131] |  |
| seed1 | 517 | 0.0178 | 0.0036 | 4.926 [3.557, 6.663] |  |
| seed2 | 517 | 0.0178 | 0.0035 | 5.111 [3.550, 7.392] |  |
| seed3 | 517 | 0.0178 | 0.0034 | 5.255 [3.703, 7.456] |  |
| seed4 | 517 | 0.0178 | 0.0035 | 5.132 [3.595, 7.406] |  |

Primary setting, other windows and the heart-rate outcome (secondary):

| outcome | click | control | estimate [95% CI] |
|---|---|---|---|
| sp3 (relative risk) | 0.0178 | 0.0033 | 5.335 [3.803, 7.500] |
| hr (mean bpm difference) | 0.4525 | -0.0028 | 0.455 [0.219, 0.809] |
| sp1 (relative risk) | 0.0079 | 0.0010 | 7.552 [6.062, 9.475] |
| sp5 (relative risk) | 0.0243 | 0.0065 | 3.730 [2.718, 5.180] |
| sp10 (relative risk) | 0.0378 | 0.0184 | 2.051 [1.646, 2.560] |
| any3 (relative risk) | 0.0246 | 0.0052 | 4.773 [3.470, 6.515] |
