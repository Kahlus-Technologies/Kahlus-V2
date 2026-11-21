# Pre-registration for K7, K1 and S1

Written 2026-11-19 by the research auditor (task `kahlus-bd-plan`, Fable 5.2), before any of these runs.
Sources: `gpu-runs-plan/report.md`, the three coder briefs (`kahlus-runner`, `kahlus-k7`, `seizure-s1`), `kahlus-bd-plan/report.md`, and the browser scans `aside-k7-novelty.md`, `aside-eeg-scaling.md` and `aside-bd-research.md`.

What I had seen when I wrote this: the dev-slice results that are already in the Kahlus-V2 repo (25 MESA + 25 SHHS-1 nights), and the seizure-alarm baseline on development patients.
What I had not seen: any K7 relative risk, any full-cohort K1 number, any S1 network result.

## 0. Rules for all three

1. **One primary cell per study.** It is named below. Every other number is secondary or exploratory and is labelled so in every table.
2. **Freeze before run.** Record the SHA-256 of this file in the PR that carries the first result. A change after a result exists goes into a "Deviations" section at the end of this file, with date and reason. Nothing above that section is edited.
3. **Development data is not confirmation.** The 50 dev-slice nights and the seizure development patients are for building and debugging. The 25 SHHS-1 dev nights (200001 to 200025) are excluded from the K7 and K1 confirmatory sets.
4. **The seals hold.** `KAHLUS_MESA` stays unset, so MESA is the 25 exposed nights. `SEIZURE_SPLIT` stays `dev`. No held-back patient file is opened, cached or copied to Engram.
5. **Intervals.** 2,000 bootstrap resamples over whole subjects or patients, percentile 95%, bootstrap seed 0.
6. **Completeness.** A summary refuses to compute if any planned job (study, model, fold, seed) has no result file. A missing fold is never averaged away.
7. **Provenance in every result file.** Git commit, config hash, data manifest hash, seed, package versions, number of starts (1 means never resumed), and the SHA-256 of this file.

## 1. K7: does the stimulation trigger aim at moments just before arousals?

**Hypothesis (two-sided).** The rate of spontaneous arousal onsets in the 3 s after a would-be click differs from the rate after matched moments with no trigger. The null is a relative risk of 1.

**Trigger (fixed, the repo's current default).** `kahlus.clas_replay.so_clicks`: causal 0.5 to 4 Hz filter on C4, crossing below -40 µV, click 0.5 s later, 2.5 s refractory, allowed only where the last completed epoch is N2 or N3 (`scripts/run_clas_replay.py`).

**Outcome.** 1 if a spontaneous arousal onset (`classify_arousals`, kind `spontaneous`) lies in (t, t + 3 s], else 0. In SHHS "spontaneous" means "not respiratory", because SHHS has no leg channel. State this next to every SHHS number.

**Matched controls (fixed here, before any K7 number).**

- For each click at time t, draw 5 control moments from the same night.
- A control is within 600 s of t, has the same stage of the last completed epoch (N2, or N3, where N3 includes stages 3 and 4), and falls in the same bin of time since the last arousal onset of any kind (under 2 min, 2 to 10 min, over 10 min).
- A control has no click within 3 s before or after it, and is at least 3 s from the other controls of the same click.
- A control time is an allowed second plus a uniform random fraction of a second.
- If fewer than 5 exist, use those that exist. If none exist, widen to 1,800 s. If still none, drop the click and count it. If more than 5% of clicks are dropped, the matching rule has failed and the result is not reported as primary.
- Control draws use seed 0. Seeds 1 to 4 are a sensitivity check.

**Primary metric.** Pooled relative risk: (bad clicks / clicks) divided by (bad controls / controls), where each click's controls have total weight 1. The interval resamples subjects.

**Primary cell.** SHHS-1, all nights that pass discovery except the 25 dev nights, default trigger, 3 s window, spontaneous arousals, matched controls as above.

**Controls of the analysis.**

1. Rotated triggers (the shift control): move each night's click times by half the night, keep those that are allowed and have no real click within 3 s, and run the same matching. The expected relative risk is 1. **Validity gate: if its interval excludes 1, the matching is biased and the primary result is not reported until that is fixed.**
2. Scorer-free outcome: heart-rate change, the mean rate in (t + 2, t + 10 s] minus the mean rate in (t - 10 s, t], from `causal_beats` with the quality mask. Report the difference between clicks and controls with a subject bootstrap. A slow wave has its own heart response, so this is a check on direction, not a second primary.
3. Sensitivity, all labelled secondary: controls drawn from any allowed moment (no "no trigger" rule), only the first click of each run of clicks, windows of 1, 5 and 10 s, any arousal, and a threshold set from a running percentile of the same night's amplitude so far (the cohorts differ in amplifier gain).

**Splits.** No model is fitted in the primary analysis, so there is no train and test split. The search over trigger settings (-40, -60, -80 µV. N2 + N3 or N3 only. Single wave or a second crossing within 2.5 s) is a selection problem. Choose the setting on the tuning subjects (the hash rule of `forecast_bench.is_tuning`, 1 subject in 5) and report its relative risk on the other subjects.

**Multiple comparisons.** One primary cell. The 11 other trigger settings use Holm. Sensitivity analyses are descriptive.

**What counts as a finding.** All four must hold: the primary interval excludes 1, the relative risk is at least 1.10 or at most 0.91, the rotated-trigger gate passes, and the two halves of SHHS-1, split by the parity of the subject ID, give the same direction with intervals that exclude 1.
The 25 MESA dev nights and the dev slice give a direction check only.

**Kill rule.** Stop the trigger-setting search if the relative risk stays between 0.9 and 1.1 for every setting on the tuning subjects. The primary cell is still reported.

**What counts as novel** (`aside-k7-novelty.md`).

- Novel: would-be click times from a causal trigger compared with matched moments inside the same sleeper, at cohort scale, and a table of trigger settings ranked by pre-arousal clicks per click. The scan found no paper with this test in 5 searches.
- Not novel: slow waves and K-complexes are part of, or come before, micro-arousals. That is in Halász 2004 and 2005 and in the cyclic alternating pattern literature, and the paper must say so and cite them.
- Must also cite: Pathak 2021 (up-state stimulation lowered noise-caused arousals) and Lechat 2021 (K-complexes are about twice as likely as arousals after noise).
- Never claim: that clicks cause or prevent arousals. A replay plays no sound.

## 2. K1: do heart features add to context at full cohort?

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

## 3. S1: a deep detector inside the seizure-alarm harness

**Hypothesis (one-sided).** At the same conformal target, a raw-signal network gives fewer false alarms than the smoothed reference detector, without catching fewer seizures.

**Baseline, rung 0 (fixed).** The repo's reference detector with scores smoothed causally: the minimum over the current and the 2 earlier windows. No other smoothing length is tried on test folds.

**Networks (fixed list).** `SPARCNet` (primary), `EEGNet`, `Deep4Net`, and the `SeizureTransformer` architecture trained from scratch. `ShallowFBCSPNet` is not run unless it replaces `Deep4Net` before any result. No pretrained weights in S1.

**Input rule.** One score every 2 s. The score for step i reads samples up to the end of the grader's 4 s window i and nothing later. Filters and normalisation are causal. No statistic of a whole recording enters the cache.

**Operating point.** The repo's `MarginalConformal` threshold at alpha 0.2, fitted on calibration patients only. "At an 80% catch rate" means this target. A threshold is never chosen from test-fold outcomes.

**Primary metric.** The repo's `median_false_alarms_per_h` at alpha 0.2 (median over patients, mean over seeds), with the pooled catch rate beside it.

**Primary cell.** CHB-MIT development patients (18), `SPARCNet` against rung 0.

**Test.** For each patient, the false alarms per hour of rung 0 minus those of the network, averaged over seeds. That gives 18 paired differences. Bootstrap over patients for the median difference. The network wins if the interval lies above 0 and its pooled catch rate is not more than 0.05 below rung 0.

**Splits.** The repo's `patient_folds` and `split_train_cal` with seeds 0, 1 and 2, unchanged. Early stopping uses 2 validation patients taken from the training patients with `numpy.random.default_rng(seed * 1000 + fold)`. Validation patients never come from the calibration set or the test fold.

**Held-back patients.** CHB-MIT chb03, chb10, chb18, chb19, chb24 and Siena PN03, PN05, PN17. The cache builder takes its patient list from `load_cohort` and a test fails if it opens one of these.

**Seeds.** 0, 1 and 2, each reported apart, as requirement R4 asks. A seed sets both the split and the weights, as the repo does today.

**Training settings.** Fixed before the grid: one learning rate, one batch size, at most 30 epochs, patience 5 on the validation loss, one class-weight rule. They are written in the PR before the first grid result.

**Secondary.** The other networks with Holm. Siena development patients (11) as a replication of the same test. Alpha 0.1 and 0.3, the worst patient, patients below target, alarm-on time, and the curve of catch rate against false alarms per hour, all descriptive. SzCORE event F1 only with a pinned scorer version.

**Kill rule.** If no network is under 10 false alarms per hour at alpha 0.2 on CHB-MIT development patients, stop adding networks.

**What counts as novel** (`aside-bd-research.md`).

- Not novel: a deep network with fewer false alarms than band power on CHB-MIT. Published numbers are far ahead on other protocols (Zabihi 2026: 0.75 sensitivity at 0.68 false alarms per hour on TUSZ. Chung 2026: 80 to 100% at 0.64 to 1.13 per hour with about 10 s latency, leave-one-patient-out).
- Novel only as part of the repo's question: how the per-patient gap (patients below target, worst patient, spread across seeds) changes when the detector improves and the calibration stays the same. The repo's reading list found no per-patient promise on both sides.
- So the S1 table must carry the per-patient columns for every model, not only the median.

## 4. Flags for firstmate to forward

Each line: what is wrong, where, and the fix. "Credibility" means the result could be wrong or unreadable. "Novelty" means the result could be already known.

1. **Credibility, K1.** No frozen plan exists in the repo. The GPU plan schedules K1 on day 3. Fix: commit section 2 of this file as `docs/results/full_cohort_plan.md`, with the captain's sign-off, before the full run.
2. **Credibility, K1.** `run_heart_arousal_benchmark.py` defaults to `--normalize none` and `--bootstrap smoke`. A run with defaults gives the wrong study. Fix: the launch command must carry `--normalize causal --bootstrap claim`, and the result file must record both.
3. **Credibility, K1 and K7.** The 25 SHHS-1 dev nights are inside SHHS-1. Every analysis choice so far was made on them. Fix: exclude 200001 to 200025 from the confirmatory sets with an ID list (`discover_nsrr` takes `ids`).
4. **Credibility, K7.** The brief says "nearby time" and "same sleep stage" and leaves the rest open. It also asks for a dev-slice table "per trigger setting" as acceptance. A coder who sees that table can pick the matching rule or the setting that looks best. Fix: implement the control rule and the primary setting of section 1 as written. The dev-slice table is a pilot, never a claim.
5. **Credibility, K7.** "then on all SHHS-1 if the dev slice works" has no definition of "works". Fix: it means the tests pass and the rotated-trigger gate passes, not that the relative risk looks interesting.
6. **Credibility, K7.** Without the rotated-trigger gate, a bias in the matching (time of night, stage mix) looks like a finding. The brief names the control but sets no pass rule. Fix: the gate of section 1.
7. **Credibility, K7.** The -40 µV threshold selects different waves in MESA and SHHS, because the repo already found that absolute EEG amplitude is a cohort fingerprint. Fix: report clicks per night per cohort, and the percentile-threshold sensitivity run.
8. **Credibility, K7.** Scorers may place an arousal onset right after a large slow wave. The heart-rate outcome does not fully separate this, because K-complexes have their own heart response. Fix: word the result as "trigger times come before scored arousals", with no mechanism.
9. **Credibility, S1.** "False alarms per hour at an 80% catch rate" can be coded as a threshold tuned on test folds until 80% are caught. That is a leak. Fix: the conformal threshold at alpha 0.2 from calibration patients, as section 3 says.
10. **Credibility, S1.** The brief names no validation patients for early stopping. The easy mistake is to stop on the calibration patients or on the test fold. Fix: the rule of section 3.
11. **Credibility, S1.** "Preprocess once into float16 arrays (fixed montage, resampled)" invites per-recording z-scores and zero-phase filters, which read the future. `SeizureTransformer`'s published recipe z-scores each channel over the whole recording. Fix: causal filters and the repo's running z-score only, with a test that a cached window does not change when later samples change.
12. **Credibility, S1.** If the cache is built straight from the PhysioNet folders on Engram, the held-back patients are in it. Fix: build the cache from `load_cohort`'s patient list, and add the test of section 3.
13. **Credibility, S1.** "ShallowFBCSPNet or Deep4Net" and "SeizureTransformer if it fits" leave the model list open until results exist. The GPU plan says 10 models, the brief names 5. Fix: the list of section 3.
14. **Credibility, S1.** Seed count differs: 3 in the brief, 5 in the GPU plan section 13. Fix: 3, the repo's protocol. Extra seeds that change only the weights are secondary.
15. **Credibility, S1.** The grader must not change in the same PR as a new detector (the repo's own rule). Fix: any change to `events.py` or `harness.py` goes in its own PR with a rerun of every method.
16. **Credibility, runner.** The brief hashes the "config dict". If the hash leaves out the code commit, the data manifest and the preprocessing settings, a stale result is skipped as "done" after a code change, and tables mix two versions of the code. Fix: all of these go into the hash, and the result file repeats them.
17. **Credibility, runner.** A job that resumes on a GPU is not bit-exact. Fix: record the number of starts in each result, state the tolerance against the seed-to-seed spread, and report whether resumed jobs differ from never-resumed jobs.
18. **Credibility, runner.** Results travel back as `RESULT` lines in logs. A cut log drops jobs without an error. Fix: rule 6 of section 0, and a count of results against the job list before any table.
19. **Credibility, all.** Early-stopping state, the best metric and the RNG states must be in the checkpoint, or a resumed run stops at a different epoch than an unbroken one. The GPU plan lists this, the runner brief does not name the best metric. Fix: add it to the acceptance test.
20. **Novelty, K7.** The slow-wave and arousal link is decades old. A write-up that presents it as the finding is not novel. Fix: the framing of section 1.
21. **Novelty, S1.** A lower false-alarm number alone is not a research result. Fix: the per-patient columns of section 3.
22. **Novelty, GPU plan E1 and K6.** `aside-eeg-scaling.md` lists a held-out-subject data-scaling study (de Oliveira and Aristimunha 2026), LaBraM at Scale with 10 encoders from 5.8M to 1.97B on up to 79,910 h, and EEG-Arena's data sweep. A family by size by data table on a few thousand open hours repeats them at a smaller scale. The scan names one open gap: a full size by data by compute law for scalp EEG on held-out subjects. Fix: E1 is worth GPU time only if it is designed to fill that gap, and its own plan already puts "unpublished" at 25%.
23. **Novelty, GPU plan S7.** EEG-Arena (30 models, 57 tasks) and NeuroAtlas (event-level epilepsy metrics) exist. Fix: the scoreboard is new only for the alarm-grade, per-patient columns.
24. **Compliance, K7 and later NSRR work.** The signed NSRR purpose is "leakage-controlled evaluation of autonomic residual forecastability for micro-arousal events". K1 fits. K7 is a stimulation replay and goes further. The GPU plan already recommends an amendment (its question Q2). A result outside the stated purpose is a publication risk. Fix: file the amendment before a K7 paper, not after.
25. **Operational.** The three briefs still say "under 4 hours and at most 40 GB per GPU", and the GPU plan says inbox 012 removed both limits. Fix: tell the coders which rule holds, because the memory cap changes how many jobs share a GPU.

## Deviations

None yet.
