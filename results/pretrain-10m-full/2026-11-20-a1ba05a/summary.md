# 10M pretraining on the full verified SHHS-1 mirror (exploratory, not a claim)

# Run summary

Infrastructure validation, not a scientific claim. This run shows that the pipeline starts, trains, writes checkpoints and scores a checkpoint on this machine. A claim needs the full cohort, a claim-mode bootstrap and multiplicity control.

- Steps logged: 50 to 4827. Loss: 2.9342 to 2.3641.
- Starts of this run: 1 (more than 1 means it was resumed).
- Last start: torch 2.9.1+cu128, CUDA 12.8, NCCL 2.27.5, 4 GPU, world size 4, commit None.
- Skill on subject_held_out (576 subjects): zero +0.0000, persistence -0.9332, ridge +0.0090, gru +0.0069, timesfm3 not run, sleepfm not run, kahlus +0.1906.

Evidence to send back: this file, `log.jsonl`, `environment.json` and `benchmark.json`. Do not send checkpoints.

The benchmark ran in smoke bootstrap mode (300 resamples) and scored 5,000 windows of 576 held-out subjects. `benchmark.json` has every model and horizon. Subject-held-out forecast skill: kahlus +0.1906 [+0.164, +0.218] against ridge +0.0090 and GRU +0.0069; shift control for kahlus -0.178.
