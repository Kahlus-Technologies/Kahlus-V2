"""Masked plus future-patch pretraining on patch shards (issue #11).

    python -m kahlus.train --shards DIR --out RUN --size 10m --train-datasets mesa mros
    torchrun --standalone --nproc-per-node=8 -m kahlus.train ...      # DDP, one process per GPU

Rerun the same command to resume: the run continues from the newest checkpoint in ``--out`` and gives
the same losses as a run that was never stopped. That holds because nothing depends on hidden state:
micro-batch i of rank r is a pure function of ``(seed, r, i)``, the learning rate is a pure function of
the step, the model has no dropout, and the checkpoint holds the model, the optimizer and the step.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from kahlus.model import SIZES, KahlusPatchModel, ModelConfig, n_params
from kahlus.nsrr import mesa_fit_allowed
from kahlus.shards import ShardSet, Windows


class MicroBatches(torch.utils.data.Dataset):
    """Item i is the whole micro-batch i of one rank: the windows and which tokens are masked."""

    def __init__(self, windows: Windows, *, total: int, batch: int, context: int, mask_ratio: float, seed: int, rank: int):
        self.windows, self.total, self.batch, self.context = windows, total, batch, context
        self.mask_ratio, self.seed, self.rank = mask_ratio, seed, rank

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng([self.seed, self.rank, i])
        x = self.windows.batch(rng, self.batch)
        masked = rng.random((self.batch, self.context, x.shape[2])) < self.mask_ratio
        return torch.from_numpy(x), torch.from_numpy(masked)


def lr_at(step: int, *, lr: float, warmup: int, steps: int) -> float:
    """Linear warm-up, then cosine decay to 10% of the peak."""
    if step < warmup:
        return lr * (step + 1) / warmup
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup))))


def latest_checkpoint(out: Path) -> Path | None:
    found = sorted(out.glob("ckpt-*.pt"))
    return found[-1] if found else None


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shards", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", choices=sorted(SIZES), required=True)
    ap.add_argument("--train-datasets", nargs="+", required=True)
    ap.add_argument("--steps", type=int, required=True, help="optimizer steps")
    ap.add_argument("--batch", type=int, default=64, help="windows per process per micro-batch")
    ap.add_argument("--accum", type=int, default=1, help="micro-batches per optimizer step")
    ap.add_argument("--context", type=int, default=64, help="seconds of context")
    ap.add_argument("--horizon", type=int, default=4, help="seconds forecast by the future head")
    ap.add_argument("--mask-ratio", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--keep", type=int, default=2, help="checkpoints kept on disk")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--workers", type=int, default=4, help="loader processes per rank")
    ap.add_argument("--amp", choices=("auto", "bf16", "off"), default="auto", help="auto: bf16 on CUDA, off on CPU")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if not mesa_fit_allowed():
        raise SystemExit("KAHLUS_MESA=final is evaluation-only: training refuses to run")

    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', 0)}" if cuda else "cpu")
    if world > 1:
        dist.init_process_group("nccl" if cuda else "gloo")
    if cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp = args.amp == "bf16" or (args.amp == "auto" and cuda)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    shards = ShardSet(args.shards)
    cfg = ModelConfig(**SIZES[args.size], n_channels=len(shards.index["channels"]), patch_len=shards.index["patch_len"],
                      context=args.context, horizon=args.horizon)
    missing = sorted(set(args.train_datasets) - {n["dataset"] for n in shards.nights})
    if missing:
        raise SystemExit(f"--train-datasets {missing} have no readable nights in {args.shards}")
    torch.manual_seed(args.seed)  # every rank builds the same initial weights
    model = KahlusPatchModel(cfg).to(device)
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), fused=cuda)

    # Everything that must match for a resumed run to equal an uninterrupted one.
    identity = {k: getattr(args, k) for k in ("size", "steps", "batch", "accum", "mask_ratio", "lr", "warmup", "weight_decay", "seed")}
    identity |= {"model": asdict(cfg), "train_datasets": sorted(args.train_datasets), "world": world, "amp": amp,
                 "shards_sha256": shards.sha256}
    start = 0
    ckpt = latest_checkpoint(out)
    if ckpt is not None:
        state = torch.load(ckpt, map_location=device, weights_only=True)
        if state["identity"] != identity:
            diff = {k: (state["identity"].get(k), v) for k, v in identity.items() if state["identity"].get(k) != v}
            raise SystemExit(f"{ckpt} was written by a different run (checkpoint, now): {diff}. Use a new --out.")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start = state["step"]
    log_path = out / "log.jsonl"
    if rank == 0:  # drop log lines past the checkpoint: the resumed run writes them again
        kept = [ln for ln in (log_path.read_text().splitlines() if log_path.exists() else []) if json.loads(ln)["step"] <= start]
        log_path.write_text("".join(ln + "\n" for ln in kept))
        print(f"{args.size}: {n_params(model):,} parameters, world={world}, device={device}, bf16={amp}, start step {start}", flush=True)

    net = torch.compile(model) if args.compile else model
    if world > 1:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device.index] if cuda else None)

    windows = Windows(shards, shards.select(args.train_datasets, heldout=False), args.context + args.horizon)
    data = MicroBatches(windows, total=args.steps * args.accum, batch=args.batch, context=args.context,
                        mask_ratio=args.mask_ratio, seed=args.seed, rank=rank)
    loader = iter(torch.utils.data.DataLoader(
        data, batch_size=None, sampler=range(start * args.accum, len(data)), num_workers=args.workers,
        pin_memory=cuda, prefetch_factor=4 if args.workers else None))

    def save(step: int) -> None:
        tmp = out / f"ckpt-{step:08d}.pt.tmp"
        torch.save({"step": step, "model": model.state_dict(), "opt": opt.state_dict(), "identity": identity}, tmp)
        os.replace(tmp, out / f"ckpt-{step:08d}.pt")  # a kill leaves the old checkpoint or the new one, never half
        for old in sorted(out.glob("ckpt-*.pt"))[: -args.keep]:
            old.unlink()

    t_data = 0.0
    t_mark, logged = time.perf_counter(), start
    for step in range(start, args.steps):
        for group in opt.param_groups:
            group["lr"] = lr_at(step, lr=args.lr, warmup=args.warmup, steps=args.steps)
        losses = torch.zeros(3, device=device)
        for k in range(args.accum):
            t0 = time.perf_counter()
            x, masked = next(loader)
            t_data += time.perf_counter() - t0
            x = x.to(device, non_blocking=True).float()
            masked = masked.to(device, non_blocking=True)
            # Ranks exchange gradients only on the last micro-batch of the step.
            sync = world == 1 or k == args.accum - 1
            with contextlib.nullcontext() if sync else net.no_sync():
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    parts = net(x, masked)
                (parts[0] / args.accum).backward()
            losses += parts.detach().float() / args.accum
        if world > 1:
            dist.all_reduce(losses)
            losses /= world
        if not torch.isfinite(losses[0]):
            raise SystemExit(f"non-finite loss at step {step + 1}; the last checkpoint in {out} is still good")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        done = step + 1
        if rank == 0 and (done % args.log_every == 0 or done == args.steps):
            wall = time.perf_counter() - t_mark
            row = {"step": done, "loss": losses[0].item(), "masked": losses[1].item(), "future": losses[2].item(),
                   "lr": opt.param_groups[0]["lr"], "data_wait_frac": round(t_data / wall, 4),
                   "patches_per_s": round((done - logged) * args.accum * args.batch * world * args.context * cfg.n_channels / wall)}
            with open(log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            t_data, t_mark, logged = 0.0, time.perf_counter(), done
        if rank == 0 and (done % args.ckpt_every == 0 or done == args.steps):
            save(done)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
