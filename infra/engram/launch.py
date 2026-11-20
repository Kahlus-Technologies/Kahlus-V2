"""Submit a job list to Engram through SkyPilot, pull the results back here, and run `sky down` when it ends.

    python infra/engram/launch.py --jobs jobs.json --name kahlus-run --gpus 8 \\
        --data-manifest MANIFEST --preprocessing PREPROC.json --prereg PREREG.md

The code that runs is `git archive HEAD`, so the recorded commit is the code by construction. Commit first.
The job hash covers the config, the commit, the data manifest hash, the preprocessing settings and the
preregistration hash, so a result from older code or data is not skipped as done. All three study files are
required, and every result file repeats them.

--jobs-per-gpu worker processes per GPU (default 4) take jobs from the list, so the whole grid runs in parallel.
At most 4 GPUs; --burst allows up to 8. Results stay on /shared
as the first copy and come back here as ``RESULT`` log lines with a digest. A line that is cut, mixed with another
or changed fails its digest, and its job counts as missing. The launcher prints how many jobs of the list have a
checked result before it prints anything else; rerun the same command and only the missing jobs are shipped.
Results go to ``results/engram/<hash>/result.json``; keep NSRR participant data out of git.
Exit code: 0 all done, 75 some missing.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SKY = str(Path.home() / ".venvs/sky/bin/sky")


def sky(*args: str, tee: Path | None = None) -> str:
    p = subprocess.Popen([SKY, *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = []
    for line in p.stdout:
        out.append(line)
        print(line, end="", flush=True)
        if tee:
            with open(tee, "a") as f:
                f.write(line)
    p.wait()
    return "".join(out)


def free_gpus() -> int:
    """Engram rule: look first. Prints `sky status` and the GPU table, returns the free H100 count."""
    sky("status")
    m = re.search(r"H100\s.*?(\d+) of (\d+) free", sky("gpus", "list", "--infra", "slurm/slinky"))
    if not m:
        sys.exit("cannot read the free GPU count from `sky gpus list`; not launching")
    return int(m.group(1))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str, text: bool = True) -> str | bytes:
    out = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=text, check=True).stdout
    return out.strip() if text else out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", required=True, type=Path)
    ap.add_argument("--name", required=True, help="SkyPilot cluster name")
    ap.add_argument("--data-manifest", required=True, type=Path, help="file listing the data used; its SHA-256 goes into the job hash")
    ap.add_argument("--preprocessing", required=True, type=Path, help="JSON file with the preprocessing settings; goes into the job hash")
    ap.add_argument("--prereg", required=True, type=Path, help="preregistration file; its SHA-256 goes into the job hash")
    ap.add_argument("--gpus", type=int, default=1, help="H100s for this cluster: 4 at most, 8 only with --burst (one node)")
    ap.add_argument("--burst", action="store_true", help="allow 5 to 8 GPUs for a short job that needs them; then drop back")
    ap.add_argument("--jobs-per-gpu", type=int, default=4, help="workers per GPU; measure jobs per hour at 1, 2, 4, 8")
    ap.add_argument("--cpus", type=int, default=8, help="total; give each worker enough for its data loading")
    ap.add_argument("--memory", type=int, default=32, help="GB")
    ap.add_argument("--disk-size", type=int, default=20, help="GB")
    ap.add_argument("--ckpt-every-s", type=float, default=900.0)
    ap.add_argument("--resume-metric", help="result key; prints resumed-vs-never-resumed gaps in seed-spread units")
    ap.add_argument("--results", type=Path, default=REPO / "results/engram")
    a = ap.parse_args()
    if not 1 <= a.gpus <= (8 if a.burst else 4):
        sys.exit("--gpus must be 1 to 4 (5 to 8 only with --burst)")
    sys.path.insert(0, str(REPO / "src"))
    from kahlus.runner import atomic_write, intake, job_hash, resume_report  # these import no torch

    jobs = json.loads(a.jobs.read_text())
    prov = {"commit": git("rev-parse", "HEAD"), "data_manifest": sha256(a.data_manifest),
            "preprocessing": json.loads(a.preprocessing.read_text()), "prereg_sha256": sha256(a.prereg)}
    want = {job_hash(j, prov): j for j in jobs}
    if len(want) != len(jobs):
        sys.exit("the job list has duplicate configs")
    todo = [j for h, j in want.items() if not (a.results / h / "result.json").exists()]

    log = a.results / f"{a.name}.log"
    if todo:
        free = free_gpus()
        if free < a.gpus:
            sys.exit(f"only {free} H100 free, need {a.gpus}; another person holds the rest. Not launching.")
        a.results.mkdir(parents=True, exist_ok=True)
        log.write_text("")
        with tempfile.TemporaryDirectory() as stage:
            archive = git("archive", "HEAD", "src", "infra/engram/setup_env.sh", "infra/engram/task.yaml", text=False)
            tarfile.open(fileobj=io.BytesIO(archive)).extractall(stage, filter="data")
            (Path(stage) / "jobs.json").write_text(json.dumps(todo))  # only the missing jobs, so no GPU waits for a done one
            (Path(stage) / "provenance.json").write_text(json.dumps(prov))
            try:
                sky("launch", "-c", a.name, "--down", "-i", "1", "-y", "--workdir", stage, "--gpus", f"H100:{a.gpus}", "--cpus", str(a.cpus),
                    "--memory", str(a.memory), "--disk-size", str(a.disk_size), "--env", f"CKPT_EVERY_S={a.ckpt_every_s}", "--env", f"JOBS_PER_GPU={a.jobs_per_gpu}",
                    str(Path(stage) / "infra/engram/task.yaml"), tee=log)
            finally:
                sky("down", "-y", a.name, tee=log)  # also on failure or Ctrl-C
        for h, payload in intake(log.read_text(), set(want)).items():
            (a.results / h).mkdir(exist_ok=True)
            atomic_write(a.results / h / "result.json", payload)

    have = {h for h in want if (a.results / h / "result.json").exists()}
    print(f"RESULTS {len(have)}/{len(want)} of the job list; missing {sorted(set(want) - have)}")
    if have == set(want) and a.resume_metric:
        print(json.dumps(resume_report([json.loads((a.results / h / "result.json").read_text()) for h in want], a.resume_metric), indent=1))
    return 0 if have == set(want) else 75


if __name__ == "__main__":
    sys.exit(main())
