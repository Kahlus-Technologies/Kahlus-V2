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
(`kahlus.model`), DDP training with bf16, gradient accumulation, periodic checkpoints and a resume that is exact on CPU
(`kahlus.train`), the benchmark harness (`kahlus.forecast_bench`) and the launcher
(`scripts/train_a100.sh`). Run instructions are in the README.

Open before the first sweep point:

- **No GPU run exists.** The CUDA, NCCL and fused-optimizer paths are written but not executed. The 2-process
  test uses the gloo backend on CPU. The cause of the 6-GPU crash in Kahlus-V1 was not reproduced.
- **MrOS is not loadable.** `kahlus.nsrr` reads MESA and SHHS only. Issue #11 trains on MESA + MrOS and tests on SHHS.
  The launcher trains on SHHS and scores held-out subjects only. Dataset-held-out scoring needs a second
  loadable dataset (`TEST_DATASET`).
- **MESA is sealed.** Without `KAHLUS_MESA=final`, the shards hold only the 25 exposed MESA nights. Pretraining
  on full MESA would break the seal in [`results/mesa_exposure.md`](results/mesa_exposure.md). This needs a decision.
- **Two channels, not ten.** The shards hold C4 EEG and ECG, the pair MESA and SHHS share.
- **TimesFM-3 and SleepFM are stubs.** The report marks them `not_run` until someone wires them in.
- **Micro-batch and learning rate per size are starting points**, not measured values.
- **No 1B run** until 10M to 200M show a slope. The launcher enforces this with `ALLOW_1B=1`.

## Training choices from the checked playbook

Each choice has a flag in `kahlus.train` and a CPU test. The arXiv IDs come from a checked literature list
(2026-11-15). The papers were not read again for this change, and no choice is measured on our data yet.

| Choice | Flag (default) | Source |
|---|---|---|
| LayerNorm on attention queries and keys, to keep attention logits bounded | `--qk-norm` (on) | arXiv 2309.14322, 2410.16682 |
| Warm-up, stable, decay learning-rate schedule; cosine stays available | `--schedule wsd` | arXiv 2410.05192 |
| Weight decay tuned together with the learning rate. Decay is the product divided by the learning rate | `--wd-lr-product 1.5e-5` | arXiv 2510.19093 says to tune them together. The fixed-product rule is ours |
| Effective batch from the data size, not the model size | `--accum` (derived) | arXiv 2410.21676 gives the direction. The 0.1%-of-windows rule is ours |
| Warning when a run reads the training data more than 4 times | `--max-epochs 4` | arXiv 2305.16264 |
| Subject-held-out splits | always on | unchanged |

Not applied:

- **z-loss** (arXiv 2309.14322) penalizes the log-partition of an output softmax. This model has regression heads and
  no output softmax, so there is nothing for it to act on. Add it if the heads change to categorical targets.

Later pilots, not in the code:

- **Muon or SOAP optimizer.** Plan for a 1.1x to 1.4x speedup, not 2x. The speedup was about 1.1x at 1.2B parameters
  (arXiv 2509.02046) and about 1.4x with well-transferred hyperparameters (arXiv 2512.05620).
- **FP8.** Only after bf16 works on the node. It matched bf16 with up to 34% more throughput, but needed
  Smooth-SwiGLU to stay stable (arXiv 2409.12517).

Two cautions for the scaling curve. Scaling-law fits depend on warm-up, compute accounting and optimizer tuning
(arXiv 2406.19146). EEG foundation models do not yet show clean scaling (arXiv 2507.11783).

## Handoff to the A100 node

The node has 8x A100 80GB. Nobody on this side has SSH to it, so a run must work as a package that its
operator starts with one command. In place: the 8-GPU default with `LOCAL_RANK` device selection, the
GPU-count preflight, the Docker fallback, `environment.json` and the scope statement in `summary.md`.
Shard `index.json` holds only dataset, subject and offsets, so it carries no label fields.

The package: `scripts/make_handoff.sh` zips a pinned commit with two checksum files. In it,
`scripts/a100_handoff.sh` runs verify, preflight, smoke, fetch (every SHHS-1 EDF), dev slice, the 10M point on
all of SHHS-1 and the evidence zip (`scripts/package_evidence.sh`, no checkpoints, data or secrets). The operator
guide is [`../README_HANDOFF.md`](../README_HANDOFF.md).

Next, not built:

- **Offline wheel set**, so that the first run needs no network for pip.
- **First run on the node.** The default image `pytorch/pytorch:2.9.1-cuda12.6-cudnn9-devel` is listed on Docker Hub
  but was not pulled or run. The host driver and the persistent path `/raid/scratch` are unchecked.

The v1 roadmap (MOABB / NeuroTwin NFC) lives in [Kahlus-V1](https://github.com/Kahlus-Technologies/Kahlus-V1).
