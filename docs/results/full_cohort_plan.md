# Full-cohort plan: K1

Frozen before the full run, signed off by the captain ("K1 approved", 2026-11-20). This is section 2 of `docs/results/k7_k1_s1_preregistration.md` (SHA-256 `9391ffad657b9395eb6ba1885291bcbe98b3755df3880de202e7a2fda6c7a94a`), copied unchanged except this header. The MESA seal code (`src/kahlus/nsrr.py`) requires this file to be tracked in git for any later `KAHLUS_MESA=final` look. Rules 1 to 7 of section 0 of the preregistration apply.

Run: `infra/engram/k1/k1_full.yaml`. The benchmark takes `--exclude-ids configs/data/shhs_dev_ids.txt` (SHHS-1 nights 200001 to 200025) and `--arousal-lag-s` for the scorer-lag control. Anchors, labels, features and the model are as at commit `9c55dad`.

## K1: do heart features add to context at full cohort?

**Hypothesis (two-sided).** Time-aligned cardiac features change the forecast of a spontaneous arousal beyond what a misaligned copy of the same features gives.

**Fixed analysis.** `scripts/run_heart_arousal_benchmark.py --normalize causal --bootstrap claim --seed 0`, with the bootstrap fix of `src/kahlus/rfs.py`. The default normaliser of that script is `none`, so the flag must be on the command line. Anchors, labels, features and the model stay as they are at commit `9c55dad`.

**Primary metric.** Aligned cardiac bits: the residual bits of `prior+cardiac` over `prior+cardiac_shift`.

**Primary cell.** SHHS-1 without the 25 dev nights, subject-grouped 5-fold cross-validation, horizon 30 s.
Why 30 s: a result counts as forecasting only at 15 s or more, and 30 s is the horizon the repo's alarm study uses.

**Secondary.** Horizons 5, 15, 60 and 120 s with Holm. The 16 beat-warning cells with a false discovery rate of 5%. Transfer from SHHS-1 to the 25 MESA dev nights, descriptive only (25 subjects).

**Controls.**

1. The shift control is inside the metric.
2. Scorer-lag control, named in the roadmap: move every arousal onset 1, 2 and 3 s earlier and rerun the primary cell. The sign must not change.
3. Seeds 1 to 4 for the model at the primary cell. The sign must not change.
4. `prior+cardiac_shift` over `prior` is reported beside the headline, to show how much the model gains from the person alone.

**Splits and held-out subjects.** `GroupKFold(5)` on the subject, as in the script. No MESA night beyond the 25 exposed ones. The one MESA look happens later, under `KAHLUS_MESA=final`, and is not part of K1.

**Decision.**

- Positive: the interval lies above 0, and controls 2 and 3 keep the sign.
- Null: the interval lies below +0.002 bits. That is 0.5% of the label entropy at 30 s (about 0.40 bits at a base rate near 8%).
- Otherwise: unresolved, and reported as such.

**Kill rule.** None. This is the measurement. A null closes the heart-only line at 30 s.

**Before the run.** The K1 text of this file is committed in Kahlus-V2 as `docs/results/full_cohort_plan.md`, signed off by the captain. The MESA seal code requires that file for any later final look.

**What counts as novel** (`aside-bd-research.md`, query 9).

- Novel: a strictly-future, shift-controlled, subject-held-out estimate in bits at about 5,700 subjects, with a null that is as publishable as a positive. The scan found "nothing in the top 5 [that] forecasts arousals ahead of onset with a lead-time metric".
- Not novel: heart rate rises at arousal onset (Huo 2023 on 1,069 MESA subjects, Azarbarzin 2014), and heart-to-brain coupling rises before arousal (Gumussu 2026, 53 participants). These must be cited as the known physiology.
- Not a K1 result: anything about detection, or about a model on raw signals. That is K2 and K3.
