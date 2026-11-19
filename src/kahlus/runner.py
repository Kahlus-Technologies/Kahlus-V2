"""Job runner: one config dict is one job. The result lands under ``<root>/<hash>/result.json`` and a job
whose result exists is skipped, so any worker can be killed and the queue just re-runs what is missing.

    python -m kahlus.runner --jobs jobs.json --provenance provenance.json --root RUNS

``jobs.json`` is a list of configs. Each has ``"fn": "package.module:function"`` and a ``"seed"``; the function
gets ``(config, checkpointer)`` and returns a JSON-able dict. ``provenance.json`` holds what else makes a
result valid: the git commit, the data manifest hash, the preprocessing settings, the preregistration hash.
The job hash covers config and provenance, so a result from older code or data is not skipped as done, and the
result file repeats the provenance. Several workers may run the same list: a job is claimed with a file lock
for as long as it runs, and the lock dies with the process.

Long jobs call ``checkpointer.step_done(...)`` after every step. It saves every ``every_s`` seconds (15 minutes)
and when SIGTERM arrived, keeps the last 2 files, and on the next start ``checkpointer.load()`` returns the
newest file that opens. SIGTERM means "yield": the worker saves at the next step boundary and exits with code
75, and a rerun resumes. The checkpoint holds the model, the optimizer, the step and the global RNG states
(torch, CUDA, NumPy, ``random``). Anything else the job owns goes into ``extra``: best metric and weights,
early-stopping counter, history, scheduler, the state of its own ``torch.Generator`` or
``numpy.random.Generator`` objects. Or make every batch a pure function of (seed, step), as ``train.py`` does.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
import platform
import random
import re
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np

YIELD_CODE = 75
stop_requested = False  # set by the SIGTERM handler that main() installs


class Yielded(Exception):
    """A stop signal arrived and the state is saved. The job is not finished; rerun it."""


def job_hash(config: dict, provenance: dict | None = None) -> str:
    """First 12 hex digits of the SHA-256 of the canonical JSON of the config and the provenance."""
    blob = json.dumps({"config": config, "provenance": provenance or {}}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def versions() -> dict:
    """Python and every installed distribution with its version."""
    v = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    return {"python": platform.python_version(), **v}


def atomic_write(path: Path, data: bytes | str | memoryview) -> None:
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


def count_start(folder: Path) -> int:
    """Number of times this job has started (1 = never resumed). Written before the work, so a kill keeps it."""
    f = folder / "starts"
    n = int(f.read_text()) + 1 if f.exists() else 1
    atomic_write(f, str(n))
    return n


def result_line(out: dict) -> str:
    """``RESULT <job> <sha256 of payload> <payload>``: one line, so a cut or interleaved line fails its digest."""
    payload = json.dumps(out, sort_keys=True)
    return f"RESULT {out['job']} {hashlib.sha256(payload.encode()).hexdigest()} {payload}"


def intake(text: str, want: set[str]) -> dict[str, str]:
    """The payload of every good RESULT line in ``text``, by job hash. A line is good when its digest matches, the
    payload parses, and the job hash is in ``want`` and is the hash of the payload's own config and provenance.
    Every other line (cut, mixed, forged, unknown job) is dropped, so its job counts as missing."""
    good = {}
    for h, digest, payload in re.findall(r"RESULT ([0-9a-f]{12}) ([0-9a-f]{64}) (\{.*\})$", text, flags=re.M):
        if hashlib.sha256(payload.encode()).hexdigest() != digest or h not in want:
            continue
        try:
            r = json.loads(payload)
            ok = r["job"] == h and job_hash(r["config"], r["provenance"]) == h
        except (ValueError, KeyError, TypeError):
            continue
        if ok:
            good[h] = payload
    return good


def resume_report(results: list[dict], metric: str) -> dict:
    """Do resumed jobs (starts > 1) differ from never-resumed ones (starts == 1)? Per config group (the config
    without its seed), the gap of each resumed result from the never-resumed mean is given in units of the
    seed-to-seed standard deviation of the never-resumed results (None when under 2 of those). Tolerance on GPU:
    a gap above 1 is not accepted, and that job is rerun unbroken."""
    groups: dict[str, list[dict]] = {}
    for r in results:
        groups.setdefault(json.dumps({k: v for k, v in r["config"].items() if k != "seed"}, sort_keys=True), []).append(r)
    out = {}
    for key, rs in groups.items():
        fresh = np.array([r["result"][metric] for r in rs if r["starts"] == 1])
        gaps = [abs(r["result"][metric] - fresh.mean()) for r in rs if r["starts"] > 1] if fresh.size else []
        spread = float(fresh.std(ddof=1)) if fresh.size > 1 else None
        ratio = [g / spread for g in gaps] if spread else None
        out[key] = {"n_never_resumed": int(fresh.size), "n_resumed": len(gaps), "seed_spread": spread,
                    "resumed_gap_over_spread": ratio, "rerun_unbroken": [x > 1 for x in ratio] if ratio else None}
    return out


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
    """Atomic checkpoints in ``directory``: ``ckpt-<step>.pt``, the newest ``keep`` kept."""

    def __init__(self, directory: Path, *, every_s: float = 900.0, keep: int = 2):
        self.dir, self.every_s, self.keep = Path(directory), every_s, keep
        self.dir.mkdir(parents=True, exist_ok=True)
        self.resumed_from: int | None = None  # step of the checkpoint this start resumed from; None for a fresh start
        self._last = self._start = time.monotonic()
        self._used_s = 0.0  # compute seconds of earlier starts, up to their last checkpoint

    def files(self) -> list[Path]:
        return sorted(self.dir.glob("ckpt-*.pt"), key=lambda p: int(p.stem.split("-")[1]))  # *.pt.tmp never matches

    def seconds(self) -> float:
        return self._used_s + time.monotonic() - self._start

    def save(self, step: int, *, model, optimizer, extra: dict | None = None) -> None:
        import torch
        state = {"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": rng_state(),
                 "extra": extra or {}, "seconds": self.seconds()}
        buf = io.BytesIO()
        torch.save(state, buf)
        atomic_write(self.dir / f"ckpt-{step:09d}.pt", buf.getbuffer())
        for old in self.files()[: -self.keep]:
            old.unlink()
        self._last = time.monotonic()

    def load(self, *, model, optimizer) -> dict | None:
        """Restore the newest checkpoint that opens, trying the older one when the newest is cut. Returns
        ``{"step", "extra"}``, or None when there is no checkpoint. Raises when files exist and none opens, so a
        real fault is never mistaken for a fresh start."""
        import torch
        files, errors = self.files(), []
        for p in reversed(files):
            try:
                s = torch.load(p, map_location="cpu", weights_only=False)
            except Exception as e:  # a cut file: try the older one
                errors.append(f"{p.name}: {e!r}")
                continue
            model.load_state_dict(s["model"])
            optimizer.load_state_dict(s["optimizer"])
            set_rng_state(s["rng"])
            self._used_s = s["seconds"]
            self.resumed_from = s["step"]
            return {"step": s["step"], "extra": s["extra"]}
        if files:
            raise RuntimeError(f"{len(files)} checkpoint file(s) in {self.dir} and none opens: {errors}")
        return None

    def step_done(self, step: int, *, model, optimizer, extra: dict | None = None) -> None:
        """Call after finishing step ``step`` (steps done so far). Saves when due or when SIGTERM arrived;
        raises ``Yielded`` after saving on SIGTERM."""
        if stop_requested or time.monotonic() - self._last >= self.every_s:
            self.save(step, model=model, optimizer=optimizer, extra=extra)
        if stop_requested:
            raise Yielded


def run_job(config: dict, root: Path, provenance: dict | None = None, *, every_s: float = 900.0) -> dict | None:
    """Run one job and return its result, or return the stored result when it exists. Returns None when another
    worker holds the job."""
    job = job_hash(config, provenance)
    folder = Path(root) / job
    result_path = folder / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return None
        if result_path.exists():  # another worker finished it between the first check and the lock
            return json.loads(result_path.read_text())
        atomic_write(folder / "config.json", json.dumps(config, sort_keys=True, indent=1))
        starts = count_start(folder)
        module, name = config["fn"].split(":")
        ck = Checkpointer(folder, every_s=every_s)
        result = getattr(importlib.import_module(module), name)(config, ck)
        out = {"job": job, "config": config, "provenance": provenance or {}, "seed": config["seed"], "versions": versions(),
               "starts": starts, "resumed_from_step": ck.resumed_from, "result": result, "seconds": round(ck.seconds(), 3)}
        atomic_write(result_path, json.dumps(out, sort_keys=True))
        return out


def work(jobs: list[dict], root: Path, provenance: dict | None = None, *, every_s: float = 900.0) -> int:
    """Run every job that has no result and that no other worker holds. Prints one RESULT line per finished job.
    Exit code: 75 after a yield, 1 when a job failed (the others still run), else 0."""
    failed = False
    for config in jobs:
        try:
            out = run_job(config, root, provenance, every_s=every_s)
        except Yielded:
            print(f"YIELDED {job_hash(config, provenance)}", flush=True)
            return YIELD_CODE
        except Exception:
            failed = True
            print(f"FAILED {job_hash(config, provenance)}\n{traceback.format_exc()}", file=sys.stderr, flush=True)
            continue
        if out is not None:
            print(result_line(out), flush=True)
    return 1 if failed else 0


def toy_train(config: dict, ck: Checkpointer) -> dict:
    """A small noisy regression with dropout and early stopping on a fixed validation batch. Batch i is a pure
    function of (seed, i); dropout reads the global RNG. The best metric, its step and the patience counter live in
    the checkpoint, so a resume that lost them or the RNG would end at another step or loss. ``sleep`` slows each
    step so that a test can stop the job mid-run."""
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
    st = prior["extra"] if prior else {"best": float("inf"), "best_step": 0, "bad": 0, "stopped": False}
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
    def on_term(*_):
        global stop_requested
        stop_requested = True

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", required=True, help="JSON file: a list of job configs")
    ap.add_argument("--provenance", required=True, help="JSON file: commit, data manifest hash, preprocessing, prereg hash")
    ap.add_argument("--root", required=True, help="results folder; one subfolder per job hash")
    ap.add_argument("--ckpt-every-s", type=float, default=900.0)
    a = ap.parse_args()
    signal.signal(signal.SIGTERM, on_term)
    sys.exit(work(json.loads(Path(a.jobs).read_text()), Path(a.root), json.loads(Path(a.provenance).read_text()), every_s=a.ckpt_every_s))


if __name__ == "__main__":
    main()
