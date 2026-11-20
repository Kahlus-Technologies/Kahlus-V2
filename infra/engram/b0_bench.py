"""B0: how fast a braindecode CNN trains on an H100 with the data in GPU memory, in a DataLoader, and in several processes.

    python b0_bench.py --out DIR

Synthetic EEG shaped like a CHB-MIT cache (18 channels, 256 Hz, 10 s windows), so no dataset is needed.
Windows per second for each model, trained with bf16 autocast and AdamW:
  loader   = torch DataLoader slicing a float16 memmap on the CPU (the plain way)
  resident = the whole float16 array on the GPU, windows cut by index arithmetic
  packed   = PROCS resident processes on one GPU at once, summed

With --mem-gb N, every process holds at most N GB of GPU memory in total. The default 0 sets no cap.
The result is written to OUT/b0_result.json and printed as one `B0_RESULT {...}` line.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch


def device_and_cap(mem_gb: float, procs: int) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    dev = torch.device("cuda", 0)
    total = torch.cuda.get_device_properties(dev).total_memory
    if mem_gb > 0:
        torch.cuda.set_per_process_memory_fraction(min(1.0, mem_gb * 2**30 / total) / procs, dev)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return dev


def build(name: str, a) -> torch.nn.Module:
    import braindecode.models as m

    return getattr(m, name)(n_chans=a.n_chans, n_outputs=2, n_times=a.win, sfreq=a.sfreq)


def make_data(a) -> Path:
    path = Path(a.out) / f"synthetic-{a.n_chans}x{a.n_samples}.f16"
    if not path.exists():
        rng = np.random.default_rng(0)
        mm = np.lib.format.open_memmap(path.with_suffix(".tmp.npy"), "w+", np.float16, (a.n_chans, a.n_samples))
        for i in range(0, a.n_samples, 1 << 22):
            mm[:, i:i + (1 << 22)] = rng.standard_normal((a.n_chans, min(1 << 22, a.n_samples - i)), np.float32)
        mm.flush()
        del mm
        os.replace(path.with_suffix(".tmp.npy"), path)
    return path


def load(path: Path) -> np.ndarray:
    return np.load(path, mmap_mode="r")


class Windows(torch.utils.data.Dataset):
    """The plain way: one window per item, sliced and z-scored on the CPU."""

    def __init__(self, path: Path, win: int, n: int, seed: int):
        self.path, self.win, self.n, self.seed, self.data = path, win, n, seed, None

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        if self.data is None:
            self.data = load(self.path)
        start = np.random.default_rng([self.seed, i]).integers(0, self.data.shape[1] - self.win)
        x = np.asarray(self.data[:, start:start + self.win], np.float32)
        x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)
        return torch.from_numpy(x), int(i % 2)


def resident_batch(data: torch.Tensor, win: int, batch: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Cut `batch` random windows from a (channels, samples) float16 tensor that lives on the device."""
    starts = torch.randint(0, data.shape[1] - win, (batch,), generator=gen, device="cpu").to(data.device)
    idx = starts[:, None] + torch.arange(win, device=data.device)
    x = data[:, idx].permute(1, 0, 2).float()  # (batch, channels, win)
    x = (x - x.mean(2, keepdim=True)) / (x.std(2, keepdim=True) + 1e-6)
    return x, (starts % 2).long()


def step(net, opt, x, y, dev) -> float:
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        loss = torch.nn.functional.cross_entropy(net(x), y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return loss.item()


def timed(fn, seconds: float, warmup: int, dev) -> float:
    """Windows per second of `fn()` (which returns the windows it trained on) after `warmup` calls."""
    for _ in range(warmup):
        fn()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        n += fn()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return n / (time.perf_counter() - t0)


def speed_one(a, model: str, mode: str, procs: int = 1) -> float:
    dev = device_and_cap(a.mem_gb, procs)
    torch.manual_seed(0)
    net = build(model, a).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    path = make_data(a)
    if mode == "loader":
        loader = iter(torch.utils.data.DataLoader(
            Windows(path, a.win, 10**9, 0), batch_size=a.batch, num_workers=a.workers,
            pin_memory=dev.type == "cuda", persistent_workers=a.workers > 0,
            prefetch_factor=4 if a.workers else None))

        def fn():
            x, y = next(loader)
            step(net, opt, x.to(dev, non_blocking=True), y.to(dev, non_blocking=True), dev)
            return len(y)
    else:
        data = torch.from_numpy(np.array(load(path))).to(dev)
        gen = torch.Generator().manual_seed(0)

        def fn():
            x, y = resident_batch(data, a.win, a.batch, gen)
            step(net, opt, x, y, dev)
            return len(y)

    return timed(fn, a.seconds, 5, dev)


def speed(a) -> dict:
    rows = {}
    for model in a.models:
        r = {mode: round(speed_one(a, model, mode)) for mode in ("loader", "resident")}
        # packed: PROCS copies of the resident case at once, each with 1/PROCS of the memory cap
        cmd = [sys.executable, __file__, *a.argv, "--child-model", model]
        kids = [subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True) for _ in range(a.procs)]
        r["packed"] = round(sum(float(k.communicate()[0].strip().splitlines()[-1]) for k in kids))
        r["resident_over_loader"] = round(r["resident"] / max(r["loader"], 1), 2)
        r["packed_over_resident"] = round(r["packed"] / max(r["resident"], 1), 2)
        rows[model] = r
        print(model, r, flush=True)
    return rows


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="EEGNet,Deep4Net,SPARCNet")
    ap.add_argument("--n-chans", type=int, default=18)
    ap.add_argument("--sfreq", type=int, default=256)
    ap.add_argument("--win-s", type=int, default=10)
    ap.add_argument("--hours", type=float, default=20.0, help="synthetic recording length")
    ap.add_argument("--batch", type=int, default=256)
    # sched_getaffinity sees the Slurm allocation; os.cpu_count() would see all 256 host threads
    ap.add_argument("--workers", type=int, default=max(1, len(os.sched_getaffinity(0)) - 2))
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=60.0, help="timed seconds per case")
    ap.add_argument("--mem-gb", type=float, default=0.0, help="GPU memory cap in GB per GPU, 0 for none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--child-model", help="internal: run one packed process and print its windows per second")
    a = ap.parse_args(argv)
    a.argv = list(argv)
    a.models = a.models.split(",")
    a.win = a.win_s * a.sfreq
    a.n_samples = int(a.hours * 3600 * a.sfreq)
    Path(a.out).mkdir(parents=True, exist_ok=True)

    if a.child_model:
        print(speed_one(a, a.child_model, "resident", a.procs), flush=True)
        return

    result = {"torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
              "cpus": len(os.sched_getaffinity(0)), "args": {k: v for k, v in vars(a).items() if k != "argv"}}
    result["speed_windows_per_s"] = speed(a)
    text = json.dumps(result, sort_keys=True)
    (Path(a.out) / "b0_result.json").write_text(text)
    print("B0_RESULT " + text, flush=True)


if __name__ == "__main__":
    main()
