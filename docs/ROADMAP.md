# Roadmap

Kahlus v2 asks one question: can heart and EEG signals tell a closed-loop sleep stimulator when *not*
to stimulate? The success measure is **wake-ups caused per 100 stimulations**.

| Phase | Goal | Exit criterion | Status |
|---|---|---|---|
| P0 | Standalone repo, CI green | Split from Kahlus-V1 with history | Done |
| P1 | Full-cohort run on SHHS-1 (~5.8k nights), scorer-lag control | Claim-mode CIs on aligned cardiac bits; plan frozen in `docs/results/full_cohort_plan.md` | Cluster run set up (`scripts/repro_dev_slice.sh`) |
| P1b | Single held-out look on MESA | One logged run with `KAHLUS_MESA=final` (see [`results/mesa_exposure.md`](results/mesa_exposure.md)) | Blocked on P1 freeze and the P02 cluster check |
| P2 | Scaling across NSRR cohorts | Same result on MrOS / CFS / SOF / STAGES / WSC | Data requests not filed |
| P3 | Hardware in the loop | Headband veto replayed on a phantom | Not started |
| P4 | Human pilot | IRB approval before any participant | Not started |
| S1 | Scaling-law sweep, 10M to 1B parameters ([#11](https://github.com/Kahlus-Technologies/Kahlus-V2/issues/11)) | A scaling curve: every size scored strictly-future, subject-held-out and dataset-held-out against the baselines | Training path proven on CPU only. No GPU run yet. Open items below |

Rules that apply to every phase: subject-held-out evaluation, shift controls on every signal claim,
dev-slice numbers are never claims, and MESA is touched once.

## Scaling sweep (S1): what exists and what is open

Exists, with tests that run on CPU: memory-mapped patch shards (`kahlus.shards`), the size presets
(`kahlus.model`), DDP training with bf16, gradient accumulation, periodic checkpoints and exact resume
(`kahlus.train`), the benchmark harness (`kahlus.forecast_bench`) and the launcher
(`scripts/train_a100.sh`). Run instructions are in the README.

Open before the first sweep point:

- **No GPU run exists.** The CUDA, NCCL and fused-optimizer paths are written but not executed. The 2-process
  test uses the gloo backend on CPU. The cause of the 6-GPU crash in Kahlus-V1 was not reproduced.
- **MrOS is not loadable.** `kahlus.nsrr` reads MESA and SHHS only. Issue #11 trains on MESA + MrOS.
- **MESA is sealed.** Without `KAHLUS_MESA=final`, the shards hold only the 25 exposed MESA nights. Pretraining
  on full MESA would break the seal in [`results/mesa_exposure.md`](results/mesa_exposure.md). This needs a decision.
- **Two channels, not ten.** The shards hold C4 EEG and ECG, the pair MESA and SHHS share.
- **TimesFM-3 and SleepFM are stubs.** The report marks them `not_run` until someone wires them in.
- **Batch size and learning rate per size are starting points**, not measured values.
- **No 1B run** until 10M to 200M show a slope. The launcher enforces this with `ALLOW_1B=1`.

The v1 roadmap (MOABB / NeuroTwin NFC) lives in [Kahlus-V1](https://github.com/Kahlus-Technologies/Kahlus-V1).
