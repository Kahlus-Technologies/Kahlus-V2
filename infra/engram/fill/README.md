# Engram fill jobs (exploratory infrastructure, not a claim)

SkyPilot tasks for the Engram Slurm cluster (`infra: slurm/slinky`). Every task uses `--down`, so a cluster ends one idle minute after its job. Code ships with the SkyPilot `workdir`. The NSRR token is a SkyPilot secret and never appears in a file, a log or git.

| Task | Hardware | What it does |
|---|---|---|
| `fill_nsrr.yaml` | 4 CPUs, no GPU | The single NSRR mirror (`nsrr_mirror.py`): SHHS-1 XMLs and EDFs into `/shared/aayu/nsrr`, 8 streams (sleepdata.org allows about 10 requests at once), MD5-verified, resumable. Prints `PROGRESS` every 2 minutes and `NSRR_MIRROR_DONE` only after the full count |
| `fill_shards.yaml` | 32 CPUs, no GPU | The mirror gate (refuses unless 5,793 EDFs and 5,793 XMLs and no `.part`), then the training shards (`kahlus.shards nsrr`) |
| `fill_a.yaml` | 4 H100, 32 CPUs | Pretraining on SHHS-1 only (MESA stays sealed), torchrun, checkpoint every N steps, resume. Default size 10M; larger sizes wait for the captain's evidence gate (`SIZES=50m`, or `GPUS=8 SIZES=200m REQUIRE_SHARDS=1` for a short burst) |
| `fill_cpu.yaml` | 64 CPUs, no GPU | The seizure-alarm CPU baseline at full scale, from a `git archive` snapshot of branch `fm/seizure-crosssite` (85cb3a2) |

## Run

```bash
infra/engram/fill/chain.sh <log-dir>      # mirror, then shards, then the 4-GPU stage, each relaunched until done
sky launch -c fill-cpu --down -y infra/engram/fill/fill_cpu.yaml   # seizure-alarm CPU benchmark
```

`fill_cpu.yaml` needs `/tmp/fm-engram-fill/sa` to hold the seizure-alarm snapshot: `git -C <seizure-alarm> archive 85cb3a2 | tar -x -C /tmp/fm-engram-fill/sa`.

Results of finished studies are in `results/`. The seizure-alarm baseline is `results/seizure-alarm-fullscale-baseline/`.
