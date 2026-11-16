"""Facts about the machine a pretraining run used, a GPU-count preflight, and the run summary.

    python -m kahlus.run_env preflight --expect-gpus 8     # fails unless exactly 8 GPUs are visible
    python -m kahlus.run_env summary RUN_DIR               # write RUN_DIR/summary.md

The person who launches a run on the A100 node is not the person who reads its results, so the run
directory must say what hardware and software produced it and what the run does and does not show.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import torch

from kahlus.nsrr_cache import git_state

SCOPE = ("Infrastructure validation, not a scientific claim. This run shows that the pipeline starts, trains, "
         "writes checkpoints and scores a checkpoint on this machine. A claim needs the full cohort, a claim-mode "
         "bootstrap and multiplicity control.")


def environment() -> dict:
    """What this process sees. Every field is read from the running process, none from configuration."""
    try:
        nccl = ".".join(map(str, torch.cuda.nccl.version()))
    except AttributeError:  # CPU builds of PyTorch carry no NCCL
        nccl = None
    return {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "nccl": nccl,
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "world_size": int(os.environ.get("WORLD_SIZE", 1)),
        "docker_image": os.environ.get("KAHLUS_DOCKER_IMAGE"),  # set by scripts/train_a100.sh inside the container
        "git": git_state(),
    }


def record(run_dir: Path, **extra) -> None:
    """Append one start of the run to ``environment.json``. A resume on a changed machine stays visible."""
    path = run_dir / "environment.json"
    starts = json.loads(path.read_text()) if path.exists() else []
    path.write_text(json.dumps([*starts, environment() | extra], indent=2) + "\n")


def summary(run_dir: Path) -> str:
    """Write ``summary.md`` from the files the run left behind, and return its text."""
    rows = [json.loads(line) for line in (run_dir / "log.jsonl").read_text().splitlines()]
    env = json.loads((run_dir / "environment.json").read_text())
    lines = ["# Run summary", "", SCOPE, "",
             f"- Steps logged: {rows[0]['step']} to {rows[-1]['step']}. Loss: {rows[0]['loss']:.4f} to {rows[-1]['loss']:.4f}.",
             f"- Starts of this run: {len(env)} (more than 1 means it was resumed).",
             f"- Last start: torch {env[-1]['torch']}, CUDA {env[-1]['torch_cuda']}, NCCL {env[-1]['nccl']}, "
             f"{env[-1]['gpu_count']} GPU, world size {env[-1]['world_size']}, commit {env[-1]['git']['commit']}."]
    bench = run_dir / "benchmark.json"
    if bench.exists():
        for split, body in json.loads(bench.read_text())["splits"].items():
            scores = ", ".join(f"{name} {m['skill']:+.4f}" if m["status"] == "ok" else f"{name} not run" for name, m in body["models"].items())
            lines.append(f"- Skill on {split} ({body['n_subjects']} subjects): {scores}.")
    else:
        lines.append("- No benchmark.json: the benchmark did not run.")
    lines += ["", "Evidence to send back: this file, `log.jsonl`, `environment.json` and `benchmark.json`. Do not send checkpoints."]
    text = "\n".join(lines) + "\n"
    (run_dir / "summary.md").write_text(text)
    return text


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--expect-gpus", type=int, required=True)
    sub.add_parser("summary").add_argument("run_dir", type=Path)
    args = ap.parse_args(argv)
    if args.command == "summary":
        print(summary(args.run_dir), end="")
        return
    env = environment()
    print(json.dumps(env, indent=2))
    if env["gpu_count"] != args.expect_gpus:
        sys.exit(f"preflight failed: {env['gpu_count']} GPU visible, {args.expect_gpus} expected "
                 f"(CUDA_VISIBLE_DEVICES={env['cuda_visible_devices']!r})")


if __name__ == "__main__":
    main()
