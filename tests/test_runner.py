"""Job runner: skip-if-done, kill-and-resume to the same stop step and best loss, SIGTERM = save then exit 75,
and provenance in the hash and the result file."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("torch")  # the `train` extra

from kahlus import runner  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
# Early stopping ends the run well before 400 steps. At 40 ms a step and a checkpoint every 0.3 s, the kill lands mid-run.
JOB = {"fn": "kahlus.runner:toy_train", "seed": 0, "steps": 400, "sleep": 0.04}
PROV = {"commit": "abc123", "data_manifest": "m1", "preprocessing": {"hz": 256}}
TOL = 1e-6  # on CPU the resume is bit-exact in practice; on GPU the documented tolerance is looser


def start(root: Path, jobs: Path) -> subprocess.Popen:
    (root.parent / "prov.json").write_text(json.dumps(PROV))
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    return subprocess.Popen([sys.executable, "-m", "kahlus.runner", "--jobs", str(jobs), "--root", str(root), "--provenance", str(root.parent / "prov.json"), "--ckpt-every-s", "0.3"],
                            env=env, stdout=subprocess.PIPE, text=True, start_new_session=True)


def wait_for_checkpoint(folder: Path, proc: subprocess.Popen) -> None:
    deadline = time.time() + 60
    while not list(folder.glob("ckpt-*.pt")):
        assert proc.poll() is None and time.time() < deadline, "no checkpoint before exit"
        time.sleep(0.02)


def folder_of(root: Path) -> Path:
    return root / runner.job_hash(JOB, PROV)


def result(root: Path) -> dict:
    return json.loads((folder_of(root) / "result.json").read_text())


def test_hash_is_stable_and_order_free():
    assert runner.job_hash({"a": 1, "b": 2}) == runner.job_hash({"b": 2, "a": 1})
    assert runner.job_hash({"a": 1}) != runner.job_hash({"a": 2})
    # prereg flag 16: new code, data or preprocessing must not be skipped as "done"
    assert runner.job_hash({"a": 1}, {"commit": "x"}) != runner.job_hash({"a": 1}, {"commit": "y"})


def test_killed_run_resumes_to_the_same_final_loss(tmp_path):
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps([JOB]))
    clean = tmp_path / "clean"
    assert start(clean, jobs).wait(timeout=120) == 0

    cut = tmp_path / "cut"
    folder = folder_of(cut)
    p = start(cut, jobs)
    wait_for_checkpoint(folder, p)
    os.killpg(p.pid, signal.SIGKILL)  # mid-run, no chance to save
    p.wait()
    assert not (folder / "result.json").exists()
    assert len(list(folder.glob("ckpt-*.pt"))) <= 2
    assert start(cut, jobs).wait(timeout=120) == 0

    ra, rb = result(clean), result(cut)
    assert (ra["starts"], rb["starts"]) == (1, 2)
    assert rb["provenance"] == PROV and rb["config_hash"] == runner.job_hash(JOB) and rb["versions"]["torch"]
    a, b = ra["result"], rb["result"]
    assert a["stop_step"] < JOB["steps"]  # early stopping ended the run, so the stop epoch is a real check (prereg flag 19)
    assert a["stop_step"] == b["stop_step"] and a["best_step"] == b["best_step"]
    assert abs(a["best_val_loss"] - b["best_val_loss"]) <= TOL


def test_sigterm_saves_and_exits_75_then_rerun_finishes(tmp_path):
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps([JOB]))
    folder = folder_of(tmp_path)
    p = start(tmp_path, jobs)
    wait_for_checkpoint(folder, p)
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=60) == runner.YIELD_CODE
    assert not (folder / "result.json").exists()
    assert start(tmp_path, jobs).wait(timeout=120) == 0
    assert result(tmp_path)["result"]["stop_step"] < JOB["steps"]


def test_finished_job_is_skipped_and_broken_newest_checkpoint_falls_back(tmp_path):
    cfg = {"fn": "kahlus.runner:toy_train", "seed": 1, "steps": 5}
    first = runner.run_job(cfg, tmp_path, PROV)
    (tmp_path / first["job"] / "result.json").write_text(json.dumps({**first, "seconds": -1}))  # marker: a rerun would overwrite it
    assert runner.run_job(cfg, tmp_path, PROV)["seconds"] == -1
    assert runner.run_job(cfg, tmp_path, {**PROV, "commit": "new"})["seconds"] != -1  # other code, so it runs again

    import torch
    m = torch.nn.Linear(2, 2)
    o = torch.optim.SGD(m.parameters(), lr=0.1)
    ck = runner.Checkpointer(tmp_path / "ck", handle_sigterm=False)
    ck.save(1, model=m, optimizer=o)
    ck.save(2, model=m, optimizer=o)
    ck.save(3, model=m, optimizer=o)
    assert [p.name for p in ck.files()] == ["ckpt-000000002.pt", "ckpt-000000003.pt"]  # keep last 2
    ck.files()[-1].write_bytes(b"truncated")
    (tmp_path / "ck" / "ckpt-000000004.pt.tmp").write_bytes(b"half written")  # a kill during a save
    assert ck.load(model=m, optimizer=o)["step"] == 2


def test_resume_report_compares_resumed_with_never_resumed_in_seed_spread_units():
    def r(seed, starts, v):
        return {"config": {"m": "x", "seed": seed}, "starts": starts, "result": {"loss": v}}
    rep = runner.resume_report([r(0, 1, 1.0), r(1, 1, 3.0), r(2, 2, 2.5)], "loss")
    (g,) = rep.values()
    assert g["n_never_resumed"] == 2 and g["n_resumed"] == 1
    assert g["resumed_gap_over_spread"] == pytest.approx([0.5 / (2 ** 0.5)])
