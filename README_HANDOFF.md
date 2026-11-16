# Kahlus-V2 A100 handoff

This package runs the Kahlus-V2 pretraining path on the Chapman 8x A100 80GB node. One script runs every stage in order. At the end it writes a small zip of results. Send that zip back.

![Handoff stages](docs/handoff_flow.svg)

(In the package, the diagram is at `kahlus-v2/docs/handoff_flow.svg`.)

## What the run proves, and what it does not

The run proves these things on this machine:

- The package arrived complete, at the commit in `COMMIT_HASH.txt`.
- PyTorch sees exactly 8 GPUs. The environment record gives their names, the driver, and the CUDA, NCCL and PyTorch versions.
- Training works on 1 GPU and across 8 GPUs (DDP, bf16, NCCL), on synthetic data and then on SHHS.
- The dev-slice studies run from start to end on a second machine. The Kahlus team compares their results with the published ones.
- A 10M-parameter model trains on SHHS for the default 4 passes over the data, writes checkpoints, and gets a benchmark score.

The run does not prove these things:

- It makes no scientific claim. Every number comes from the 25-night dev slice. `summary.md` says this too.
- The 10M run is not a real sweep point. On 25 SHHS nights, 4 passes is only about 20 optimizer steps. It tests the 8-GPU path from start to end, nothing more.
- It does not test the 50M, 200M and 1B sizes.
- It does not show whether a resumed GPU run gives the same losses as an uninterrupted run. That is tested on CPU only.

## What you need

- Linux with the NVIDIA driver, 8 A100 GPUs, `bash`, `curl`, `python3` and `sha256sum`.
- One of these two:
  - Python 3.10 or newer with the `venv` module (on Ubuntu: `python3-venv`), or
  - Docker with the NVIDIA container toolkit (`docker run --gpus all` must work).
- Network access. pip installs packages, the fetch downloads from sleepdata.org, and Docker pulls its image.
- Your own sleepdata.org token in `~/.nsrr_token`. The token stays on your machine. The package does not contain a token, and the results zip never takes one.
- Free disk for the NSRR data (about 7 GB), the Python environment with CUDA PyTorch (several GB), and the Docker image if you use Docker.

## Run it

1. Unpack the zip and check it. `unzip` works too.

   ```bash
   python3 -m zipfile -e kahlus-a100-handoff-<sha>.zip .
   cd kahlus-a100-handoff-<sha>
   sha256sum -c SHA256SUMS
   tar xzf kahlus-v2-<sha>.tar.gz
   ```

2. Run every stage:

   ```bash
   kahlus-v2/scripts/a100_handoff.sh
   ```

   If no Python 3.10+ with `venv` is available, use Docker. Every Python stage then runs in `pytorch/pytorch:2.9.1-cuda12.6-cudnn9-devel`:

   ```bash
   DOCKER=1 kahlus-v2/scripts/a100_handoff.sh
   ```

3. Send back the file that the last line names: `kahlus-evidence-<sha>.zip`.

If a stage fails, the script still writes the evidence zip and tells you its path. Send it back as it is. After a fix, run the same command again. The script skips every stage whose outputs already verify.

Outputs go to `WORK`. The default is `/raid/scratch/$USER/kahlus-a100` if the script can make that folder, else `~/kahlus-a100`. To change it, set `WORK=/some/path` before the command.

## The stages

To run only some stages, name them. The script still runs them in the order below.

```bash
kahlus-v2/scripts/a100_handoff.sh verify preflight smoke
```

| Stage | What it does | Output in `WORK` | Skipped when |
| --- | --- | --- | --- |
| `verify` | Checks `SHA256SUMS` (the package files), `HANDOFF_SHA256SUMS` (every repo file), that no file changed since checkout, and the commit | `verify.txt` | never |
| `preflight` | Stops unless PyTorch sees exactly 8 GPUs (`NGPU`). Records the GPUs and versions. Builds the Python environment the first time | `preflight/environment.json`, `preflight/nvidia-smi.txt` | never |
| `smoke` | Tiny model on synthetic data: 200 steps on 1 GPU and a benchmark, then 20 steps on all 8 GPUs | `smoke/run/summary.md`, `smoke/ddp/` | `smoke/run/summary.md` and `benchmark.json` exist |
| `fetch` | Downloads all MESA and SHHS-1 annotations and 25 EDFs of each, with your `~/.nsrr_token` (about 7 GB) | `nsrr/` | every file on the fetch lists is present |
| `devslice` | Runs every dev-slice study (`scripts/repro_dev_slice.sh`) | `devslice/handoff-<sha>/` | its `run_end.json` says `ok` |
| `train10m` | Builds SHHS patch shards, then runs `scripts/train_a100.sh 10m`: preflight, 8-GPU training, benchmark, summary | `shards/`, `runs/kahlus-10m/` | `runs/kahlus-10m/summary.md` and `benchmark.json` exist |
| `evidence` | Packs the results zip | `kahlus-evidence-<sha>.zip` | never |

A stage is skipped only if it finished for the same commit. Each stage writes its log to `logs/<stage>.log`.

If you already have the NSRR data in the same layout as the fetch (one folder per cohort, with all files in it), set `NSRR_ROOT` to that folder and leave out `fetch`.

Other settings:

- `NGPU` (default 8): the GPUs that the preflight expects and that training uses. Use `CUDA_VISIBLE_DEVICES` to hide GPUs.
- `JOBS`, `THREADS`, `BOOTSTRAP` go to `scripts/repro_dev_slice.sh`. `STEPS` and `WORKERS` go to `scripts/train_a100.sh`. Their headers explain them.

## The Docker path

With `DOCKER=1`, each Python stage runs in a new container. The container:

- runs as your user, so all files in `WORK` stay yours,
- sees all GPUs (`--gpus all --ipc=host`), and `CUDA_VISIBLE_DEVICES` selects among them,
- mounts only the repo, `WORK` and `NSRR_ROOT`, at the same paths as on the host,
- keeps its Python packages in `WORK/venv-docker`, so pip runs only the first time.

The fetch and the checksum checks run on the host, so your token never goes into a container. Set `DOCKER_IMAGE` to use a different image.

## Connect-back (optional, one time)

This step lets the Kahlus team reach the node directly, so you do not have to send every result by hand. It is not necessary for the run. Do it only if you agree to it.

1. The person who owns the Tailscale account makes a one-time auth key and gives it to you.
2. Put the key in a file, and then run:

   ```bash
   kahlus-v2/scripts/a100_connect_back.sh --auth-key-file ~/ts-key.txt
   rm ~/ts-key.txt
   ```

The script installs Rex in `~/.local/bin` and runs it as your user. It needs no root access and opens no ports. Only the Tailscale account that owns the key can connect. To see each command before it runs, add `--dry-run`. The repo README tells more.

## What you send back

The evidence zip contains only text records from `WORK`:

- `verify.txt`, `preflight/environment.json`, `preflight/nvidia-smi.txt`
- `smoke/run/` and `runs/kahlus-10m/`: `summary.md`, `log.jsonl`, `environment.json`, `benchmark.json`
- `devslice/handoff-<sha>/`: `run.json`, `run_end.json`, `steps.tsv`, the study results, the discovery manifests and the logs
- `logs/<stage>.log` for each stage
- `MANIFEST.sha256`, with the SHA-256 of each file in the zip
- `EXCLUDED.txt`, with each file that the zip does not contain and the reason

The zip never contains checkpoints, NSRR data, shards, feature caches, Python environments, tokens, keys or `.env` files. It also leaves out each file that contains the text of `~/.nsrr_token` or a private key. It takes only `.json`, `.jsonl`, `.md`, `.txt`, `.tsv`, `.log` and `.csv` files of 20 MB or less. Before you send the zip, you can look at its file list with `unzip -l`.

## Known limits

- Nobody has run this package on the A100 node. The CUDA, NCCL and Docker paths have run only in tests on CPU.
- The first run needs network access. There is no offline wheel set yet.
- The default Docker image is listed on Docker Hub, but nobody has pulled or run it on the node.
- `/raid/scratch` comes from an old checklist. If the script cannot make its folder there, it uses `~/kahlus-a100`.
- MrOS cannot be loaded, and MESA is sealed. Thus the 10M run trains on SHHS and gets a score only on held-out SHHS subjects.
