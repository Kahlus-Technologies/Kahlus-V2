"""Run command lines on GPUs, at most --gpus at once, one GPU per command (CUDA_VISIBLE_DEVICES). A command that exits non-zero is retried
twice. Commands are idempotent (they exit at once when their done.json exists), so a relaunch after a stop continues the grid.

    python scripts/e1_queue.py jobs.txt --gpus 4
"""
import argparse
import os
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--retries", type=int, default=2)
    a = ap.parse_args()
    todo = [(c.strip(), 0) for c in open(a.jobs) if c.strip() and not c.startswith("#")]
    free, running, failed = list(range(a.gpus)), {}, []
    while todo or running:
        while todo and free:
            cmd, tries = todo.pop(0)
            g = free.pop(0)
            running[subprocess.Popen(cmd, shell=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g)})] = (g, cmd, tries)
            print(f"start gpu{g}: {cmd[:120]}", flush=True)
        time.sleep(5)
        for p in [p for p in running if p.poll() is not None]:
            g, cmd, tries = running.pop(p)
            free.append(g)
            if p.returncode and tries < a.retries:
                todo.append((cmd, tries + 1))
            elif p.returncode:
                failed.append(cmd)
            print(f"done gpu{g} rc={p.returncode}: {cmd[:120]}", flush=True)
    if failed:
        print("FAILED", *failed, sep="\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
