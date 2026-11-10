"""scripts/repro_dev_slice.sh: fresh run dirs, fail-fast, claim bootstrap and causal normalization."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/repro_dev_slice.sh"

# Stands in for python: logs each call, makes --out, writes a subset file for nsrr_discover
# --limit, fails the step named by STUB_FAIL and makes the step named by STUB_SLOW hang.
STUB = r"""#!/usr/bin/env bash
set -eu
name=$(basename "$1" .py)
printf '%s|%s|ids=%s|omp=%s\n' "$name" "$*" "${NSRR_SHHS_IDS:-}" "${OMP_NUM_THREADS:-}" >> "$STUB_LOG"
out="" prev=""
for a in "$@"; do [[ $prev == --out ]] && out=$a; prev=$a; done
[[ -n $out ]] && mkdir -p "$out"
if [[ $name == nsrr_discover && " $* " == *" --limit "* ]]; then printf '200001\n' > "$out/ids_shhs.txt"; fi
[[ $name == "${STUB_FAIL:-none}" ]] && exit 3
if [[ $name == "${STUB_SLOW:-none}" ]]; then sleep 20; touch "$STUB_LOG.slow_finished"; fi
exit 0
"""


@pytest.fixture
def env(tmp_path):
    stub = tmp_path / "py"
    stub.write_text(STUB)
    stub.chmod(0o755)
    for d in ("mesa", "shhs", "runs"):
        (tmp_path / d).mkdir()
    e = {k: v for k, v in os.environ.items() if not k.startswith(("NSRR_", "KAHLUS_", "STEPS", "JOBS", "THREADS", "SHHS_"))}
    return e | {"PYTHON": str(stub), "STUB_LOG": str(tmp_path / "calls.log"), "OUT_BASE": str(tmp_path / "runs"),
                "NSRR_MESA_ROOT": str(tmp_path / "mesa"), "NSRR_SHHS_ROOT": str(tmp_path / "shhs")}


def _run(env: dict, *args: str, bash: str = "bash", **extra) -> subprocess.CompletedProcess:
    return subprocess.run([bash, str(SCRIPT), *args], env=env | extra, capture_output=True, text=True, timeout=120)


def _calls(env: dict) -> list[list[str]]:
    return [line.split("|") for line in Path(env["STUB_LOG"]).read_text().splitlines()]


def test_full_run_uses_claim_bootstrap_causal_norm_and_a_fresh_dir(env):
    r = _run(env, "r1", SHHS_LIMIT="3", THREADS="2")
    assert r.returncode == 0, r.stderr
    calls = _calls(env)
    names = [c[0] for c in calls]
    assert names == ["nsrr_discover", "nsrr_discover", "nsrr_arousal_census", "run_heart_arousal_benchmark",
                     "run_beat_warning", "run_clas_replay", "run_montage_stress", "event_locked_hr",
                     "run_conformal_alarm", "run_conformal_alarm"]
    args = {c[0]: c[1] for c in calls}
    assert "--bootstrap claim" in args["run_heart_arousal_benchmark"] and "--normalize causal" in args["run_heart_arousal_benchmark"]
    assert "--bootstrap claim" in args["run_beat_warning"]
    run = Path(env["OUT_BASE"]) / "r1"
    assert f"--cache {run}/bench/cache" in args["run_heart_arousal_benchmark"]
    assert all(c[2] == f"ids={run}/discovery/ids_shhs.txt" for c in calls[2:])  # every study sees the subset
    assert all(c[3] == "omp=2" for c in calls[2:])
    meta = json.loads((run / "run.json").read_text())
    assert (meta["bootstrap"], meta["shhs_limit"], meta["normalize"]) == ("claim", "3", "causal")
    assert json.loads((run / "run_end.json").read_text())["status"] == "ok"
    rows = (run / "steps.tsv").read_text().splitlines()
    assert len(rows) == 9 and all(row.split("\t")[1] == "0" for row in rows[1:])

    again = _run(env, "r1")  # an existing run dir is never reused
    assert again.returncode == 2 and "already exists" in again.stderr


def test_a_failed_step_stops_the_run(env):
    r = _run(env, "r2", STUB_FAIL="run_beat_warning", BOOTSTRAP="smoke")
    assert r.returncode == 1
    names = [c[0] for c in _calls(env)]
    assert names[-1] == "run_beat_warning" and "run_conformal_alarm" not in names
    assert "--bootstrap smoke" in _calls(env)[-1][1]
    end = json.loads((Path(env["OUT_BASE"]) / "r2/run_end.json").read_text())
    assert (end["status"], end["failed_step"]) == ("failed", "beat")


@pytest.mark.parametrize("extra, msg", [
    ({"KAHLUS_MESA": "final"}, "dev mode only"),
    ({"BOOTSTRAP": "fast"}, "claim or smoke"),
    ({"STEPS": "conf"}, "must include bench"),
    ({"STEPS": "bench nope"}, "unknown step"),
    ({"NSRR_SHHS_ROOT": "/nonexistent"}, "not a directory"),
])
def test_bad_settings_refuse_before_any_work(env, extra, msg):
    r = _run(env, "r3", **extra)
    assert r.returncode == 2 and msg in r.stderr
    assert not (Path(env["OUT_BASE"]) / "r3").exists() and not Path(env["STUB_LOG"]).exists()


def _bash_with_wait_n() -> str | None:
    for b in (shutil.which("bash"), "/opt/homebrew/bin/bash", "/usr/local/bin/bash", "/bin/bash"):
        if b and Path(b).exists():
            v = subprocess.run([b, "-c", "echo $((BASH_VERSINFO[0]*100+BASH_VERSINFO[1]))"], capture_output=True, text=True)
            if v.returncode == 0 and int(v.stdout) >= 403:
                return b
    return None


@pytest.mark.skipif(_bash_with_wait_n() is None, reason="JOBS>1 needs bash 4.3+")
def test_parallel_failure_stops_the_other_steps(env):
    t0 = time.monotonic()
    r = _run(env, "r4", bash=_bash_with_wait_n(), JOBS="3", STEPS="mont census beat",
             STUB_FAIL="run_beat_warning", STUB_SLOW="run_montage_stress")
    assert r.returncode == 1 and time.monotonic() - t0 < 15
    end = json.loads((Path(env["OUT_BASE"]) / "r4/run_end.json").read_text())
    assert end["failed_step"] == "beat"
    time.sleep(1)
    assert not Path(env["STUB_LOG"] + ".slow_finished").exists()
