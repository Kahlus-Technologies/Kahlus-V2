# Roadmap

Kahlus v2 asks one question: can heart and EEG signals tell a closed-loop sleep stimulator when *not*
to stimulate? The success measure is **wake-ups caused per 100 stimulations**.

| Phase | Goal | Exit criterion | Status |
|---|---|---|---|
| P0 | Standalone repo, CI green | Split from Kahlus-V1 with history | Done |
| P1 | Full-cohort run on SHHS-1 (~5.8k nights), scorer-lag control | Claim-mode CIs on aligned cardiac bits; plan frozen in `docs/results/full_cohort_plan.md` | Cluster run set up (`v2/full-cohort-run`) |
| P1b | Single held-out look on MESA | One logged run with `KAHLUS_MESA=final` (see [`results/mesa_exposure.md`](results/mesa_exposure.md)) | Blocked on P1 freeze and the P02 cluster check |
| P2 | Scaling across NSRR cohorts | Same result on MrOS / CFS / SOF / STAGES / WSC | Data requests not filed |
| P3 | Hardware in the loop | Headband veto replayed on a phantom | Not started |
| P4 | Human pilot | IRB approval before any participant | Not started |

Rules that apply to every phase: subject-held-out evaluation, shift controls on every signal claim,
dev-slice numbers are never claims, and MESA is touched once.

The v1 roadmap (MOABB / NeuroTwin NFC) lives in [Kahlus-V1](https://github.com/Kahlus-Technologies/Kahlus-V1).
