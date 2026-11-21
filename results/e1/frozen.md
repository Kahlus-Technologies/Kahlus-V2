# E1 frozen settings (step 8 pilots, r1, seed 0; docs/prereg/e1.md)

Choices use the lowest tune-subject log loss on the primary readout (detect) of a probe with C = 0.1. The mask is chosen at 10M and kept at 50M and 200M.

| Arm | Mask | lr 10M | lr 50M | lr 200M | Tune loss at 10M (random weights 0.3620) | s per step 10M / 50M / 200M | Collapsed pilots |
|---|---|---|---|---|---|---|---|
| signal | 0.3 | 0.0003 | 0.0003 | 0.0003 | 0.2523 | 0.073 / 0.200 / 0.563 | 0 of 6 |
| latent | 0.6 | 0.001 | 0.001 | 0.0003 | 0.2500 | 0.106 / 0.288 / 0.787 | 0 of 6 |

K1 (collapse): not triggered, latent_failed = False.
K3 (budget): projected 148.7 GPU-hours for the full grid (cap 900), cuts: none. The projection adds 8% for embeddings and probes and uses the measured seconds per step above.
Compute match: a latent step costs 1.45 times a signal step at 10M (the design assumed 1.33), so with 14,400 against 19,200 steps the latent arm spends about 9% more GPU time than the signal arm. The report states measured GPU-seconds for both.
