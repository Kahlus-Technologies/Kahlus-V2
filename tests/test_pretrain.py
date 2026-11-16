"""Pretraining path of issue #11, proven on CPU: causal shards, the MESA seal, 1- and 2-process (gloo DDP)
runs, kill-and-resume with identical losses, and the benchmark harness on the resulting tiny checkpoint."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # the `train` extra; the rest of the suite runs without it

from kahlus import forecast_bench, nsrr, shards, train  # noqa: E402
from kahlus.model import SIZES, KahlusPatchModel, ModelConfig, n_params  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
STEPS = 120
TINY = ["--size", "tiny", "--train-datasets", "synth_a", "--steps", str(STEPS), "--batch", "8", "--accum", "2",
        "--context", "8", "--horizon", "2", "--warmup", "10", "--ckpt-every", "20", "--log-every", "1", "--workers", "0"]


def test_causal_resample_reads_no_future():
    x = np.random.default_rng(0).normal(size=2560)  # 10 s at 256 Hz
    y = np.array(x)
    y[1280:] += 1.0  # change everything from second 5 on
    a, b = shards.causal_resample(x, 256, 125), shards.causal_resample(y, 256, 125)
    assert a.size == 1250
    assert np.array_equal(a[:625], b[:625]) and not np.allclose(a[700:], b[700:])


def test_causal_patches_read_no_future():
    x = np.random.default_rng(0).normal(size=(2, 100 * shards.PATCH_LEN))
    y = np.array(x)
    y[:, 80 * shards.PATCH_LEN :] *= 5.0
    a, b = shards.causal_patches(x), shards.causal_patches(y)
    kept = 80 - shards.WARMUP_PATCHES
    assert a.shape == (100 - shards.WARMUP_PATCHES, 2, shards.PATCH_LEN) and a.dtype == np.float16
    assert np.array_equal(a[:kept], b[:kept]) and not np.array_equal(a[kept:], b[kept:])


def test_model_never_reads_later_seconds():
    torch.manual_seed(0)
    model = KahlusPatchModel(ModelConfig(**SIZES["tiny"], patch_len=16, context=8, horizon=2))
    x = torch.randn(2, 8, 2, 16)
    y = x.clone()
    y[:, 5:] += 1.0
    with torch.no_grad():
        for a, b in zip(model.predict(x), model.predict(y)):
            assert torch.allclose(a[:, :5], b[:, :5], atol=1e-6) and not torch.allclose(a[:, 5:], b[:, 5:])


@pytest.mark.parametrize("name, target", [("10m", 10e6), ("50m", 50e6), ("200m", 200e6), ("1b", 1e9)])
def test_size_presets_match_their_names(name, target):
    # Depth adds the same parameters per layer, so two shallow builds give the full count without the memory.
    one, two = (n_params(KahlusPatchModel(ModelConfig(**SIZES[name] | {"n_layers": k}))) for k in (1, 2))
    total = one + (SIZES[name]["n_layers"] - 1) * (two - one)
    assert 0.9 * target < total < 1.1 * target


def test_shards_keep_the_mesa_seal(tmp_path, monkeypatch):
    monkeypatch.delenv("KAHLUS_MESA", raising=False)
    exposed = sorted(nsrr.mesa_exposed_ids())[0]
    night = np.zeros((40, 2, 4), dtype=np.float16)
    shards.write_shards([("mesa", exposed, night), ("mesa", "9999", night), ("shhs", "200001", night)], tmp_path)
    seen = {(n["dataset"], n["subject"]) for n in shards.ShardSet(tmp_path).nights}
    assert seen == {("mesa", exposed), ("shhs", "200001")}

    monkeypatch.setenv("KAHLUS_MESA", "final")
    with pytest.raises(SystemExit, match="evaluation-only"):
        train.main(["--shards", str(tmp_path), "--out", str(tmp_path / "run"), "--size", "tiny", "--train-datasets", "shhs", "--steps", "1"])
    assert not (tmp_path / "run").exists()


def test_nsrr_edf_becomes_a_shard(tmp_path, monkeypatch):
    import mne

    monkeypatch.delenv("KAHLUS_MESA", raising=False)
    subject = sorted(nsrr.mesa_exposed_ids())[0]
    root = tmp_path / "mesa"
    root.mkdir()
    data = np.random.default_rng(0).normal(0, 1e-5, size=(2, 256 * 180))  # 3 min at the MESA rate
    raw = mne.io.RawArray(data, mne.create_info(["EEG3", "EKG"], 256.0, ch_types=["eeg", "ecg"]), verbose=False)
    mne.export.export_raw(root / f"mesa-sleep-{subject}.edf", raw, fmt="edf", overwrite=True, verbose=False)
    (root / f"mesa-sleep-{subject}-nsrr.xml").write_text("<PSGAnnotation><ScoredEvents/></PSGAnnotation>")
    monkeypatch.setenv("NSRR_MESA_ROOT", str(root))

    shards.main(["nsrr", "--cohorts", "mesa", "--out", str(tmp_path / "out")])
    (night,) = shards.ShardSet(tmp_path / "out").nights
    assert (night["dataset"], night["subject"], night["n"]) == ("mesa", subject, 180 - shards.WARMUP_PATCHES)


@pytest.fixture(scope="module")
def shard_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("shards")
    shards.write_shards(shards.synthetic_nights(), out, shard_patches=1500)  # several shards, as on the cluster
    return out


def _start(shard_dir, out, nproc):
    launch = ["-m", "kahlus.train"] if nproc == 1 else ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={nproc}", "-m", "kahlus.train"]
    env = {k: v for k, v in os.environ.items() if k != "KAHLUS_MESA"} | {"OMP_NUM_THREADS": "1"}
    return subprocess.Popen([sys.executable, *launch, "--shards", str(shard_dir), "--out", str(out), *TINY],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def _finish(proc):
    output, _ = proc.communicate(timeout=300)
    assert proc.returncode == 0, output


def _losses(out):
    return [json.loads(line)["loss"] for line in (out / "log.jsonl").read_text().splitlines()]


@pytest.fixture(scope="module")
def reference_run(shard_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("ref1")
    _finish(_start(shard_dir, out, 1))
    return out


@pytest.mark.parametrize("nproc", [1, 2])
def test_kill_and_resume_reproduces_the_uninterrupted_run(shard_dir, tmp_path, reference_run, nproc):
    ref = reference_run
    if nproc > 1:
        ref = tmp_path / "ref"
        _finish(_start(shard_dir, ref, nproc))
    assert len(_losses(ref)) == STEPS and _losses(ref)[-1] < _losses(ref)[0]

    out = tmp_path / "killed"
    proc = _start(shard_dir, out, nproc)
    deadline = time.monotonic() + 120
    while not list(out.glob("ckpt-*.pt")) and proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    # SIGKILL every process of the run at once, like a node failure. torchrun puts its workers in their own
    # sessions, so killing the launcher alone would leave them running.
    subprocess.run(["pkill", "-KILL", "-f", str(out)], check=True)
    proc.communicate()
    at_kill = int(train.latest_checkpoint(out).stem.split("-")[1])
    assert proc.returncode != 0 and at_kill < STEPS, "the run finished before the kill; nothing was tested"

    _finish(_start(shard_dir, out, nproc))
    assert _losses(out) == _losses(ref)
    a, b = (torch.load(train.latest_checkpoint(d), weights_only=True)["model"] for d in (out, ref))
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_resume_refuses_a_different_run(shard_dir, reference_run, monkeypatch):
    monkeypatch.delenv("KAHLUS_MESA", raising=False)
    args = ["--shards", str(shard_dir), "--out", str(reference_run), *TINY]
    with pytest.raises(SystemExit, match="different run"):
        train.main([*args, "--lr", "1e-3"])


def test_benchmark_scores_the_tiny_checkpoint(shard_dir, reference_run, tmp_path, monkeypatch):
    monkeypatch.delenv("KAHLUS_MESA", raising=False)
    args = ["--ckpt", str(train.latest_checkpoint(reference_run)), "--shards", str(shard_dir), "--out", str(tmp_path / "report.json"),
            "--n-fit", "400", "--n-eval", "200", "--gru-steps", "100", "--gru-hidden", "16"]
    report = forecast_bench.main([*args, "--test-dataset", "synth_b"])
    assert json.loads((tmp_path / "report.json").read_text()) == report
    assert report["checkpoint"]["train_datasets"] == ["synth_a"] and report["test_dataset"] == "synth_b"
    for split in ("subject_held_out", "dataset_held_out"):
        models = report["splits"][split]["models"]
        assert report["splits"][split]["n_subjects"] >= 2
        assert models["zero"]["skill"] == 0.0
        # The synthetic signal repeats every second, so persistence must score well when aligned and lose it when shifted.
        assert models["persistence"]["skill"] > 0.3 and models["persistence"]["gain_over_shift"] > 0.3
        assert models["persistence"]["skill_ci_low"] <= models["persistence"]["skill"] <= models["persistence"]["skill_ci_high"]
        for name in ("ridge", "gru", "kahlus"):
            assert models[name]["status"] == "ok" and np.isfinite(models[name]["skill"])
        for name in ("timesfm3", "sleepfm"):
            assert models[name]["status"] == "not_run" and models[name]["reason"]

    with pytest.raises(SystemExit, match="training dataset"):
        forecast_bench.main([*args, "--test-dataset", "synth_a"])


def test_launch_script_prints_its_commands():
    def dry(size, **env):
        return subprocess.run([str(REPO / "scripts/train_a100.sh"), size, "/data/shards", "/data/runs"],
                              env=os.environ | {"DRY_RUN": "1", "NGPU": "8"} | env, capture_output=True, text=True)

    ok = dry("200m")
    assert ok.returncode == 0, ok.stderr
    assert "--nproc-per-node=8 -m kahlus.train" in ok.stdout and "--size 200m" in ok.stdout and "--amp bf16" in ok.stdout
    assert "kahlus.forecast_bench" in ok.stdout and "--test-dataset shhs" in ok.stdout
    assert dry("1b").returncode == 2 and dry("1b", ALLOW_1B="1").returncode == 0
