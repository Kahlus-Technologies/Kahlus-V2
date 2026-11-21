# E1 gates (docs/prereg/e1.md, sha256 68bbda211b5b810629389d3155740c3f399f244566779c5f31dbf4633d38fdcb)

## GQ, quality gate: PASS (2026-11-21, CPU job e1-cpu)
SHHS-1 shards `shhs1-full`, pool P = 5,194 nights (non-held-out, non-dev), 43,751 h.

| Cell | Nights | Hours | Median bad fraction |
|---|---|---|---|
| r1 | 5,194 | 43,751 | 0.0582 |
| r4 | 1,302 | 10,975 | 0.0602 |
| r16 | 352 | 2,966 | 0.0640 |
| clean4 | 1,296 | 10,825 | 0.0086 |
| dirty4 | 1,296 | 11,040 | 0.6929 |

dirty4 median bad fraction is 80.2 times that of clean4 (difference 0.684), so GQ (3 times and 0.01) passes. The dirty4 value is high: those nights are mostly
flat, railed or burst patches by this metric, so the quality contrast is large.

## Anchors (step 6): PASS
574 test subjects (is_heldout), 978,519 anchors in total, 40,005 detect positives in test (needs 400 subjects and 2,000 positives). Fit 300 and tune 100 subjects.

## G0, range gate: PASS, primary readout = detect (2026-11-21, job e1-g0, 1 H100)
Checkpoint: newest `ckpt-00004827.pt` of the full-data 10M run. Test = 574 held-out subjects. Bits are test log loss differences per subject, paired subject bootstrap (2,000 resamples, seed 0).

| Readout | Pretrained minus random weights (bits, 3 seeds) | 95% interval | Over band power | Over base rate |
|---|---|---|---|---|
| detect | +0.1244 | [+0.1144, +0.1346] | +0.1234 [+0.1123, +0.1354] | +0.1588 |
| stage | pass (interval above 0) | | | +0.8498 |

The range is large (the pilot A 30 s forecast had +0.0014 bits in total), so `detect` is the primary and `stage` the secondary. Band power does not beat the encoder here (K4 is not triggered by G0).
