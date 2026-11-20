# Pilots A and B on partial SHHS-1 shards (exploratory, not a claim)

Test subjects: 190 (held out of pretraining). Bits = log-loss difference / ln 2, positive favours the first arm. Paired subject bootstrap, 2000 resamples, seed 0.

## Pilot A

| Cell | Mean bits | 95% interval | Subjects | Holm p |
|---|---|---|---|---|
| fine-tune pretrained minus scratch, 30 s (primary) | +0.0000 | [-0.0003, +0.0004] | 190 | nan |
| fine-tune pretrained minus scratch, 5 s | +0.0000 | [-0.0001, +0.0002] | 190 | 1.000 |
| fine-tune pretrained minus scratch, 15 s | +0.0000 | [-0.0002, +0.0003] | 190 | 1.000 |
| fine-tune pretrained minus scratch, 60 s | -0.0001 | [-0.0007, +0.0005] | 190 | 1.000 |
| fine-tune pretrained minus scratch, 120 s | -0.0002 | [-0.0011, +0.0005] | 190 | 1.000 |
| probe pretrained minus random-init, 30 s | +0.0003 | [-0.0011, +0.0013] | 190 | 1.000 |
| probe pretrained minus prior + band power, 30 s | -0.0025 | [-0.0039, -0.0013] | 190 | 0.007 |
| probe per-window input minus as-is input, 30 s | -0.0003 | [-0.0018, +0.0012] | 190 | 1.000 |

Shift control (pretrained fine-tune on half-night-shifted windows, bits over the prior-only logistic, 30 s): -0.0002 [-0.0005, +0.0001]

Fine-tune pretrained, bits over the prior-only logistic at 30 s: +0.0014 [+0.0004, +0.0025]

Fine-tune random, bits over the prior-only logistic at 30 s: +0.0013 [+0.0003, +0.0024]

**Pilot A verdict: no clear gain.** Gate for larger sizes: the primary interval above 0, mean at least +0.002 bits, shift control passes.

## Pilot B

| Feature set | Subject-ID acc (chance 0.005) | Amplitude acc (chance 0.2) | Stage balanced acc (chance 0.2) |
|---|---|---|---|
| emb-b-pretrained-asis-s0 | 0.2439 | 0.4320 | 0.4969 |
| emb-b-pretrained-perwindow-s0 | 0.2060 | 0.4035 | 0.4616 |
| emb-b-random-asis-s0 | 0.0707 | 0.3769 | 0.4137 |
| emb-b-random-asis-s1 | 0.0686 | 0.3737 | 0.4091 |
| emb-b-random-asis-s2 | 0.0706 | 0.3700 | 0.4190 |
| emb-b-random-perwindow-s0 | 0.0510 | 0.3410 | 0.3441 |
| emb-b-random-perwindow-s1 | 0.0512 | 0.3348 | 0.3405 |
| emb-b-random-perwindow-s2 | 0.0524 | 0.3300 | 0.3492 |
| bandpower-b-asis-s0 | 0.0575 | 0.4169 | 0.5284 |
| bandpower-b-perwindow-s0 | 0.0483 | 0.3723 | 0.4602 |

Subject-ID accuracy, pretrained minus random-init (as-is, 190 subjects): +0.1739 [+0.1520, +0.1956]; ratio 3.49; stage balanced accuracy gain +0.0830. **Fingerprint finding: no.**

Amplitude probe, pretrained: as-is 0.432, per-window 0.403. **Amplitude finding: no.**
