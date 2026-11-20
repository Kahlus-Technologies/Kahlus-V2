"""Job runner: kill and resume, SIGTERM then exit 75, what a checkpoint must hold, and the per-job claim."""
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")  # the `train` extra

from kahlus import runner  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
# Early stopping ends the run well before 400 steps. At 40 ms a step and a checkpoint every 0.3 s, a kill lands mid-run.
JOB = {"fn": "kahlus.runner:toy_train", "seed": 0, "steps": 400, "sleep": 0.04}
PROV = {"commit": "abc123", "data_manifest": "m1", "preprocessing": {"hz": 256}, "prereg_sha256": "p1"}
TOL = 1e-6  # on CPU a resume is bit-exact in practice; on GPU it is only approximate


def start(root: Path, jobs: Path) -> subprocess.Popen:
    (root.parent / "prov.json").write_text(json.dumps(PROV))
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    cmd = [sys.executable, "-m", "kahlus.runner", "--jobs", str(jobs), "--root", str(root),
           "--provenance", str(root.parent / "prov.json"), "--ckpt-every-s", "0.3"]
    return subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, text=True, start_new_session=True)


def folder_of(root: Path) -> Path:
    return root / runner.job_hash(JOB, PROV)


def wait_for_checkpoint(folder: Path, proc: subprocess.Popen) -> None:
    deadline = time.time() + 60
    while not list(folder.glob("ckpt-*.pt")):
        assert proc.poll() is None and time.time() < deadline, "no checkpoint before exit"
        time.sleep(0.02)


def result(root: Path) -> dict:
    return json.loads((folder_of(root) / "result.json").read_text())


class Halt(runner.Checkpointer):
    """Saves every step and stops the job at step ``at``, as a kill right after a save would."""
    at = 30

    def step_done(self, step, **state):
        self.save(step, **state)
        if step == self.at:
            raise runner.Yielded


def halted_then_resumed(folder: Path, at: int, tamper=None):
    cfg = {"seed": 0, "steps": 400}
    Halt.at = at
    with pytest.raises(runner.Yielded):
        runner.toy_train(cfg, Halt(folder))
    if tamper:
        path = runner.Checkpointer(folder).files()[-1]
        state = torch.load(path, weights_only=False)
        tamper(state)
        torch.save(state, path)
    ck = runner.Checkpointer(folder)
    return runner.toy_train(cfg, ck), ck


def test_hash_is_stable_and_order_free():
    assert runner.job_hash({"a": 1, "b": 2}) == runner.job_hash({"b": 2, "a": 1})
    assert runner.job_hash({"a": 1}) != runner.job_hash({"a": 2})
    # new code, data or preprocessing must not be skipped as done
    assert runner.job_hash({"a": 1}, {"commit": "x"}) != runner.job_hash({"a": 1}, {"commit": "y"})


def test_killed_run_resumes_to_the_same_stop_step_and_best_loss(tmp_path):
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps([JOB]))
    clean = tmp_path / "clean"
    assert start(clean, jobs).wait(timeout=120) == 0

    cut = tmp_path / "cut"
    p = start(cut, jobs)
    wait_for_checkpoint(folder_of(cut), p)
    os.killpg(p.pid, signal.SIGKILL)  # mid-run, no chance to save
    p.wait()
    assert not (folder_of(cut) / "result.json").exists()
    assert len(list(folder_of(cut).glob("ckpt-*.pt"))) <= 2
    assert start(cut, jobs).wait(timeout=120) == 0

    ra, rb = result(clean), result(cut)
    assert (ra["starts"], ra["resumed_from_step"]) == (1, None)
    assert rb["starts"] == 2 and 0 < rb["resumed_from_step"] < rb["result"]["stop_step"]  # it resumed, it did not restart
    assert rb["provenance"] == PROV and rb["versions"]["torch"]
    a, b = ra["result"], rb["result"]
    assert a["stop_step"] < JOB["steps"]  # early stopping ended the run, so the stop step is a real check
    assert a["stop_step"] == b["stop_step"] and a["best_step"] == b["best_step"]
    assert abs(a["best_val_loss"] - b["best_val_loss"]) <= TOL


def test_sigterm_saves_and_exits_75_then_rerun_resumes(tmp_path):
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps([JOB]))
    folder = folder_of(tmp_path)
    p = start(tmp_path, jobs)
    wait_for_checkpoint(folder, p)
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=60) == runner.YIELD_CODE
    assert not (folder / "result.json").exists()
    assert start(tmp_path, jobs).wait(timeout=120) == 0
    r = result(tmp_path)
    assert r["starts"] == 2 and r["resumed_from_step"] > 0 and r["result"]["stop_step"] < JOB["steps"]


def test_resume_needs_the_extra_state_and_the_rng_state(tmp_path):
    clean = runner.toy_train({"seed": 0, "steps": 400}, runner.Checkpointer(tmp_path / "clean"))
    # seed 0 stops at step 100 with its best loss at step 85
    ok, ck = halted_then_resumed(tmp_path / "ok", 90)
    assert ck.resumed_from == 90 and ok == clean

    def no_extra(state):  # best metric and patience counter lost
        state["extra"] = {"best": float("inf"), "best_step": 0, "bad": 0, "stopped": False}

    def no_rng(state):  # RNG state lost: dropout masks differ
        torch.manual_seed(12345)
        state["rng"]["torch"] = torch.get_rng_state()

    assert halted_then_resumed(tmp_path / "e", 90, no_extra)[0] != clean
    assert halted_then_resumed(tmp_path / "r", 30, no_rng)[0] != clean


def test_finished_job_is_skipped_and_new_provenance_runs_again(tmp_path):
    cfg = {"fn": "kahlus.runner:toy_train", "seed": 1, "steps": 5}
    first = runner.run_job(cfg, tmp_path, PROV)
    (tmp_path / first["job"] / "result.json").write_text(json.dumps({**first, "seconds": -1}))  # marker: a rerun would overwrite it
    assert runner.run_job(cfg, tmp_path, PROV)["seconds"] == -1
    assert runner.run_job(cfg, tmp_path, {**PROV, "commit": "new"})["seconds"] != -1


def test_job_without_a_seed_fails(tmp_path):
    with pytest.raises(KeyError):
        runner.run_job({"fn": "kahlus.runner:toy_train", "steps": 5}, tmp_path, PROV)


def test_keeps_two_checkpoints_and_loads_the_older_when_the_newest_is_cut(tmp_path):
    m = torch.nn.Linear(2, 2)
    o = torch.optim.SGD(m.parameters(), lr=0.1)
    ck = runner.Checkpointer(tmp_path)
    for step in (1, 2, 3):
        ck.save(step, model=m, optimizer=o)
    assert [p.name for p in ck.files()] == ["ckpt-000000002.pt", "ckpt-000000003.pt"]
    ck.files()[-1].write_bytes(b"cut")
    (tmp_path / "ckpt-000000004.pt.tmp").write_bytes(b"half written")  # a kill during a save
    assert ck.load(model=m, optimizer=o)["step"] == 2 and ck.resumed_from == 2


def test_load_raises_when_checkpoints_exist_and_none_opens(tmp_path):
    m = torch.nn.Linear(2, 2)
    o = torch.optim.SGD(m.parameters(), lr=0.1)
    (tmp_path / "ckpt-000000001.pt").write_bytes(b"cut")
    with pytest.raises(RuntimeError, match="none opens"):
        runner.Checkpointer(tmp_path).load(model=m, optimizer=o)
    assert runner.Checkpointer(tmp_path / "empty").load(model=m, optimizer=o) is None


def test_a_job_held_by_another_worker_is_left_to_it(tmp_path):
    cfg = {"fn": "kahlus.runner:toy_train", "seed": 2, "steps": 5}
    folder = tmp_path / runner.job_hash(cfg, PROV)
    folder.mkdir()
    with open(folder / "lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        assert runner.run_job(cfg, tmp_path, PROV) is None
    assert runner.run_job(cfg, tmp_path, PROV)["starts"] == 1


def test_resume_report_compares_resumed_with_never_resumed_in_seed_spread_units():
    def r(seed, starts, v):
        return {"config": {"m": "x", "seed": seed}, "starts": starts, "result": {"loss": v}}
    (g,) = runner.resume_report([r(0, 1, 1.0), r(1, 1, 3.0), r(2, 2, 2.5), r(3, 2, 5.0)], "loss").values()
    assert g["n_never_resumed"] == 2 and g["n_resumed"] == 2
    assert g["resumed_gap_over_spread"] == pytest.approx([0.5 / 2 ** 0.5, 3.0 / 2 ** 0.5])
    assert g["rerun_unbroken"] == [False, True]
    json.dumps(g)  # the launcher prints it
