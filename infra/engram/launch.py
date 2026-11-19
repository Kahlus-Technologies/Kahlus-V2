"""Submit a job list to Engram through SkyPilot, pull the results back here, and always run `sky down`.

    python infra/engram/launch.py --jobs jobs.json --name kahlus-toy --gpus 1 [--cpus 8 --memory 32 --disk-size 20]

The job hash covers the config, the git commit, the data manifest hash, the preprocessing settings and the
preregistration hash (prereg flag 16), so older results never count as done. The launcher refuses a dirty tree.
Results go to ``results/engram/<hash>/result.json`` (aggregate numbers only; nothing from NSRR goes in git).
They also stay on /shared as the first copy. Here they come back as ``RESULT`` log lines; the launcher checks each
one (its hash must match its own config and provenance) and counts them against the job list before it prints
any number (prereg rule 6, flag 18). The remote worker skips jobs that already have a result on /shared, so
rerun the same command after a yield or a failure and only the missing jobs run.
Exit code: 0 all done, 75 some missing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SKY = os.environ.get("SKY", str(Path.home() / ".venvs/sky/bin/sky"))
RESULT = re.compile(r"RESULT ([0-9a-f]{12}) (\{.*\})")


def sky(*args: str, tee: Path | None = None, check: bool = True) -> str:
    p = subprocess.Popen([SKY, *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = []
    for line in p.stdout:
        out.append(line)
        print(line, end="", flush=True)
        if tee:
            with open(tee, "a") as f:
                f.write(line)
    if p.wait() and check:
        print(f"sky {args[0]} exited {p.returncode}", file=sys.stderr)
    return "".join(out)


def free_gpus() -> int:
    """Engram rule: look first. Prints `sky status` and the GPU table, returns the free H100 count."""
    sky("status", check=False)
    m = re.search(r"H100\s.*?(\d+) of (\d+) free", sky("gpus", "list", "--infra", "slurm/slinky", check=False))
    if not m:
        sys.exit("cannot read the free GPU count from `sky gpus list`; not launching")
    return int(m.group(1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", required=True, type=Path)
    ap.add_argument("--name", required=True, help="SkyPilot cluster name")
    ap.add_argument("--gpus", type=int, default=1, help="H100s for this cluster, 8 at most (one node)")
    ap.add_argument("--cpus", type=int, default=8)
    ap.add_argument("--memory", type=int, default=32, help="GB")
    ap.add_argument("--disk-size", type=int, default=20, help="GB")
    ap.add_argument("--data-manifest", type=Path, help="file listing the data used; its SHA-256 goes into the job hash")
    ap.add_argument("--preprocessing", type=Path, help="JSON file with the preprocessing settings; goes into the job hash")
    ap.add_argument("--prereg", type=Path, help="preregistration file; its SHA-256 goes into the job hash")
    ap.add_argument("--resume-metric", help="result key; prints resumed-vs-never-resumed gaps in seed-spread units (prereg flag 17)")
    ap.add_argument("--allow-dirty", action="store_true", help="toy runs only; the result then records dirty=true")
    ap.add_argument("--results", type=Path, default=REPO / "results/engram")
    a = ap.parse_args()
    jobs = json.loads(a.jobs.read_text())
    if not 1 <= a.gpus <= 8:
        sys.exit("--gpus must be 1 to 8")
    sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest() if f else None  # noqa: E731
    git = lambda *c: subprocess.run(["git", "-C", str(REPO), *c], capture_output=True, text=True, check=True).stdout.strip()  # noqa: E731
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    if dirty and not a.allow_dirty:
        sys.exit("working tree has uncommitted changes; commit first so the result names its code (or --allow-dirty for a toy)")
    prov = {"commit": git("rev-parse", "HEAD"), "dirty": dirty, "data_manifest": sha(a.data_manifest),
            "preprocessing": json.loads(a.preprocessing.read_text()) if a.preprocessing else None, "prereg_sha256": sha(a.prereg)}
    free = free_gpus()
    if free < a.gpus:
        sys.exit(f"only {free} H100 free, need {a.gpus}; another person holds the rest. Not launching.")

    a.results.mkdir(parents=True, exist_ok=True)
    log = a.results / f"{a.name}.log"
    log.write_text("")
    with tempfile.TemporaryDirectory() as stage:
        shutil.copytree(REPO / "src", Path(stage) / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
        shutil.copy(HERE / "setup_env.sh", stage)
        shutil.copy(a.jobs, Path(stage) / "jobs.json")
        (Path(stage) / "provenance.json").write_text(json.dumps(prov))
        try:
            sky("launch", "-c", a.name, "--down", "-y", "--workdir", stage, "--gpus", f"H100:{a.gpus}", "--cpus", str(a.cpus),
                "--memory", str(a.memory), "--disk-size", str(a.disk_size), str(HERE / "task.yaml"), tee=log, check=False)
        finally:
            sky("down", "-y", a.name, tee=log, check=False)  # even on failure or Ctrl-C

    from kahlus.runner import atomic_write, job_hash, resume_report  # torch is not imported by these
    want = {job_hash(j, prov): j for j in jobs}
    for h, blob in RESULT.findall(log.read_text()):
        r = json.loads(blob)
        if h in want and r["job"] == h and job_hash(r["config"], r["provenance"]) == h and r["provenance"] == prov:
            (a.results / h).mkdir(exist_ok=True)
            atomic_write(a.results / h / "result.json", blob)
    have = {h for h in want if (a.results / h / "result.json").exists()}
    print(f"RESULTS {len(have)}/{len(want)} of the job list; missing {sorted(set(want) - have)}")
    if have == set(want) and a.resume_metric:
        print(json.dumps(resume_report([json.loads((a.results / h / "result.json").read_text()) for h in want], a.resume_metric), indent=1))
    return 0 if have == set(want) else 75


if __name__ == "__main__":
    sys.path.insert(0, str(REPO / "src"))
    sys.exit(main())
