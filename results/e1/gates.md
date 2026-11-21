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
