"""Job runner: one config dict is one job. The result lands under ``<root>/<hash>/result.json`` and a job
whose result exists is skipped, so any worker can be killed and the queue just re-runs what is missing.

    python -m kahlus.runner --jobs jobs.json --root RUNS [--rank R --workers N]

``jobs.json`` is a list of configs. ``--provenance FILE`` holds what else makes a result valid: the code commit,
the data manifest hash, the preprocessing settings, the preregistration hash. It goes into the job hash, so a
result from older code or data is never skipped as "done", and each result file repeats it (prereg rule 7).
 Each has ``"fn": "package.module:function"``; the function gets
``(config, checkpointer)`` and returns a JSON-able dict. Long jobs call ``checkpointer.step_done(...)`` after
every step. It saves every ``every_s`` seconds (15 minutes) and on SIGTERM, keeps the last 2 files, and on
the next start ``checkpointer.load()`` returns the newest file that opens. SIGTERM means "yield": the worker
saves at the next step boundary and exits with code 75, and a rerun resumes.

The learning-rate schedule should be a pure function of the step (as in ``train.py``), so the checkpoint does
not carry a scheduler. On CPU a resumed run is bit-exact. On GPU it is only approximate.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np

YIELD_CODE = 75  # torch is imported inside the functions that need it, so launch.py can import job_hash without it


class Yielded(Exception):
    """A stop signal arrived and the state is saved. The job is not finished; rerun it."""


def job_hash(config: dict, provenance: dict | None = None) -> str:
    """First 12 hex digits of the SHA-256 of the canonical JSON of the config and the provenance."""
    blob = json.dumps({"config": config, "provenance": provenance or {}}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def versions() -> dict:
    import importlib.metadata as md
    import platform
    v = {"python": platform.python_version()}
    for pkg in ("torch", "numpy", "braindecode"):
        try:
            v[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    return v


def count_start(folder: Path) -> int:
    """Number of times this job has started (1 = never resumed). Survives kills: written before the work."""
    f = folder / "starts"
    n = int(f.read_text()) + 1 if f.exists() else 1
    atomic_write(f, str(n))
    return n


def resume_report(results: list[dict], metric: str) -> dict:
    """Prereg flag 17: do resumed jobs (starts > 1) differ from never-resumed ones (starts == 1)? Per config
    group (the config without its seed), the gap of each resumed result from the never-resumed mean is given in
    units of the seed-to-seed standard deviation of the never-resumed results (None when under 2 of those)."""
    groups: dict[str, list[dict]] = {}
    for r in results:
        groups.setdefault(json.dumps({k: v for k, v in r["config"].items() if k != "seed"}, sort_keys=True), []).append(r)
    out = {}
    for key, rs in groups.items():
        fresh = np.array([r["result"][metric] for r in rs if r["starts"] == 1])
        gaps = [abs(r["result"][metric] - fresh.mean()) for r in rs if r["starts"] > 1] if fresh.size else []
        spread = float(fresh.std(ddof=1)) if fresh.size > 1 else None
        out[key] = {"n_never_resumed": int(fresh.size), "n_resumed": len(gaps), "seed_spread": spread,
                    "resumed_gap_over_spread": [g / spread for g in gaps] if spread else None}
    return out


def atomic_write(path: Path, data: bytes | str) -> None:
    """Write, fsync, rename, fsync the folder: a reader sees the old file or the whole new one."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data.encode() if isinstance(data, str) else data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def rng_state() -> dict:
    import torch
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy": np.random.get_state(), "python": random.getstate()}


def set_rng_state(s: dict) -> None:
    import torch
    torch.set_rng_state(s["torch"])
    if s["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])
    np.random.set_state(s["numpy"])
    random.setstate(s["python"])


class Checkpointer:
    """Atomic checkpoints in ``directory``: ``ckpt-<step>.pt``, newest ``keep`` kept."""

    def __init__(self, directory: Path, *, every_s: float = 900.0, keep: int = 2, handle_sigterm: bool = True):
        self.dir, self.every_s, self.keep = Path(directory), every_s, keep
        self.dir.mkdir(parents=True, exist_ok=True)
        self.stop = False
        self._last = self._start = time.monotonic()
        self._used_s = 0.0  # wall time of earlier runs of this job, for GPU-hour accounting
        if handle_sigterm:
            signal.signal(signal.SIGTERM, lambda *_: setattr(self, "stop", True))

    def files(self) -> list[Path]:
        return sorted(self.dir.glob("ckpt-*.pt"), key=lambda p: int(p.stem.split("-")[1]))  # *.pt.tmp never matches

    def save(self, step: int, *, model, optimizer, extra: dict | None = None) -> None:
        import torch
        state = {"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": rng_state(),
                 "extra": extra or {}, "wall_s": self._used_s + time.monotonic() - self._start}
        tmp = self.dir / f"ckpt-{step:09d}.pt"
        torch.save(state, str(tmp) + ".tmp")
        with open(str(tmp) + ".tmp", "rb+") as f:
            os.fsync(f.fileno())
        os.replace(str(tmp) + ".tmp", tmp)
        for old in self.files()[: -self.keep]:
            old.unlink()
        self._last = time.monotonic()

    def load(self, *, model, optimizer) -> dict | None:
        """Restore the newest checkpoint that opens, trying the older one when the newest is broken.
        Returns ``{"step", "extra"}``, or None for a fresh start."""
        import torch
        for p in reversed(self.files()):
            try:
                s = torch.load(p, map_location="cpu", weights_only=False)
            except Exception as e:  # truncated or corrupt: fall back to the older file
                print(f"checkpoint {p.name} unreadable ({e!r}); trying the older one", file=sys.stderr)
                continue
            model.load_state_dict(s["model"])
            optimizer.load_state_dict(s["optimizer"])
            set_rng_state(s["rng"])
            self._used_s = s["wall_s"]
            return {"step": s["step"], "extra": s["extra"]}
        return None

    def step_done(self, step: int, *, model, optimizer, extra: dict | None = None) -> None:
        """Call after finishing step ``step`` (steps done so far). Saves when due or on a stop signal;
        raises ``Yielded`` after saving on a stop signal."""
        if self.stop or time.monotonic() - self._last >= self.every_s:
            self.save(step, model=model, optimizer=optimizer, extra=extra)
        if self.stop:
            raise Yielded


def run_job(config: dict, root: Path, provenance: dict | None = None, *, every_s: float = 900.0) -> dict:
    """Run one job, or return the stored result when it exists."""
    job = job_hash(config, provenance)
    folder = Path(root) / job
    result_path = folder / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    folder.mkdir(parents=True, exist_ok=True)
    atomic_write(folder / "config.json", json.dumps(config, sort_keys=True, indent=1))
    starts = count_start(folder)
    module, name = config["fn"].split(":")
    fn = getattr(importlib.import_module(module), name)
    t0 = time.monotonic()
    ck = Checkpointer(folder, every_s=every_s)
    out = {"job": job, "config": config, "config_hash": job_hash(config), "provenance": provenance or {}, "seed": config.get("seed"),
           "versions": versions(), "starts": starts, "result": fn(config, ck)}
    out["seconds"] = round(time.monotonic() - t0, 3)
    atomic_write(result_path, json.dumps(out, sort_keys=True))
    return out


def work(jobs: list[dict], root: Path, provenance: dict | None = None, *, rank: int = 0, workers: int = 1, every_s: float = 900.0) -> int:
    """Run jobs ``rank, rank+workers, ...``. Prints ``RESULT <hash> <json>`` per finished job.
    Exit code: 75 after a yield, 1 when a job failed (the others still run), else 0."""
    failed = False
    for i, config in enumerate(jobs):
        if i % workers != rank:
            continue
        try:
            out = run_job(config, root, provenance, every_s=every_s)
        except Yielded:
            print(f"YIELDED {job_hash(config, provenance)}", flush=True)
            return YIELD_CODE
        except Exception:
            failed = True
            print(f"FAILED {job_hash(config, provenance)}\n{traceback.format_exc()}", file=sys.stderr, flush=True)
            continue
        print(f"RESULT {out['job']} {json.dumps(out, sort_keys=True)}", flush=True)
    return 1 if failed else 0


def toy_train(config: dict, ck: Checkpointer) -> dict:
    """A small noisy regression with dropout and early stopping on a fixed validation batch. Batch i is a pure
    function of (seed, i); dropout reads the global RNG. The best metric, its step and the patience counter live in
    the checkpoint, so a resume that lost them or the RNG would stop at another step. ``sleep`` slows each step so
    a test can kill the job mid-run."""
    import torch
    torch.manual_seed(config["seed"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(32, 1)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

    def batch(seed: int, n: int):
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(n, 16, generator=g)
        w = torch.randn(16, 1, generator=torch.Generator().manual_seed(config["seed"] + 1))
        return x.to(dev), (x @ w + 0.5 * torch.randn(n, 1, generator=g)).to(dev)

    xv, yv = batch(-config["seed"] - 1, 256)
    prior = ck.load(model=model, optimizer=opt)
    st = prior["extra"] if prior else {"step": 0, "best": float("inf"), "best_step": 0, "bad": 0, "stopped": False}
    step = prior["step"] if prior else 0
    while step < config["steps"] and not st["stopped"]:
        x, y = batch(config["seed"] * 1_000_003 + step, 64)
        loss = torch.nn.functional.mse_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        if step % config.get("eval_every", 5) == 0:
            model.eval()
            with torch.no_grad():
                v = torch.nn.functional.mse_loss(model(xv), yv).item()
            model.train()
            if v < st["best"] - 1e-3:
                st.update(best=v, best_step=step, bad=0)
            else:
                st["bad"] += 1
            st["stopped"] = st["bad"] >= config.get("patience", 3)
        time.sleep(config.get("sleep", 0))
        ck.step_done(step, model=model, optimizer=opt, extra=st)
    return {"stop_step": step, "best_val_loss": st["best"], "best_step": st["best_step"], "device": dev}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", required=True, help="JSON file: a list of job configs")
    ap.add_argument("--root", required=True, help="results folder; one subfolder per job hash")
    ap.add_argument("--provenance", help="JSON file: commit, data manifest hash, preprocessing, prereg hash")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--ckpt-every-s", type=float, default=900.0)
    a = ap.parse_args()
    prov = json.loads(Path(a.provenance).read_text()) if a.provenance else {}
    sys.exit(work(json.loads(Path(a.jobs).read_text()), Path(a.root), prov, rank=a.rank, workers=a.workers, every_s=a.ckpt_every_s))


if __name__ == "__main__":
    main()
