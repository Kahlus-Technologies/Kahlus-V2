"""Pretraining path of issue #11, proven on CPU: causal shards, the MESA seal, 1- and 2-process (gloo DDP)
runs, kill-and-resume with identical losses, and the benchmark harness on the resulting tiny checkpoint."""
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # the `train` extra; the rest of the suite runs without it

from kahlus import forecast_bench, nsrr, run_env, shards, train  # noqa: E402
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
    assert [s["start_step"] for s in json.loads((out / "environment.json").read_text())] == [0, at_kill]


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

    only_subjects = forecast_bench.main(args)
    assert list(only_subjects["splits"]) == ["subject_held_out"] and only_subjects["test_dataset"] is None


def test_launch_script_prints_its_commands(tmp_path):
    def dry(size, **env):
        clean = {k: v for k, v in os.environ.items() if k not in ("NGPU", "DOCKER", "DOCKER_IMAGE", "TEST_DATASET", "TRAIN_DATASETS")}
        return subprocess.run([str(REPO / "scripts/train_a100.sh"), size, "/data/shards", "/data/runs"],
                              env=clean | {"DRY_RUN": "1"} | env, capture_output=True, text=True)

    ok = dry("200m")
    assert ok.returncode == 0, ok.stderr
    steps = [ln for ln in ok.stdout.splitlines() if ln.startswith("+ ")]
    assert [next(m for m in ("run_env preflight", "kahlus.train", "forecast_bench", "run_env summary") if m in ln) for ln in steps] == [
        "run_env preflight", "kahlus.train", "forecast_bench", "run_env summary"]
    assert "--expect-gpus 8" in steps[0]  # 8 GPUs unless NGPU says otherwise
    assert "--nproc_per_node=8 -m kahlus.train" in steps[1] and "--size 200m" in steps[1] and "--amp bf16" in steps[1]
    assert "--steps" not in steps[1] and "--steps 500" in dry("200m", STEPS="500").stdout  # no step count unless STEPS sets one
    assert "--accum" not in steps[1]  # the trainer derives the effective batch from the data size
    assert "--test-dataset" not in ok.stdout
    assert "--test-dataset shhs" in dry("200m", TEST_DATASET="shhs").stdout
    assert "--expect-gpus 4" in dry("200m", NGPU="4").stdout and "--nproc_per_node=4 " in dry("200m", NGPU="4").stdout

    docker = dry("200m", DOCKER="1")
    (line,) = [ln for ln in docker.stdout.splitlines() if ln.startswith("+ ")]  # the container reruns this script, nothing else runs on the host
    assert line.startswith("+ docker run") and "--gpus all" in line and "--ipc=host" in line and "/data/runs:/data/runs" in line
    assert "200m /data/shards /data/runs" in line and "device=" not in line
    assert "GIT_CONFIG_KEY_0=safe.directory" in line and "GIT_CONFIG_VALUE_0=/workspace/repo" in line  # root in the container reads the host-owned repo

    # The default datasets must have readable training nights in shards written for the NSRR cohorts, and MESA stays sealed.
    train_line = next(ln for ln in ok.stdout.splitlines() if "kahlus.train" in ln).split()
    first = train_line.index("--train-datasets") + 1
    default_datasets = train_line[first : next(i for i in range(first, len(train_line)) if train_line[i].startswith("-"))]
    assert "mesa" not in default_datasets
    shard_root = tmp_path / "default-shards"
    shards.write_shards(shards.synthetic_nights(datasets=tuple(nsrr.NSRR_COHORTS)), shard_root)
    assert shards.ShardSet(shard_root).select(default_datasets, heldout=False)
    assert dry("1b").returncode == 2 and dry("1b", ALLOW_1B="1").returncode == 0


def test_qk_norm_removes_the_scale_of_queries_and_keys():
    # Oracle: LayerNorm(a * x) equals LayerNorm(x) for a > 0. So with QK-layernorm (arXiv 2309.14322) the attention
    # weights cannot grow with the size of the query and key projections. Without it they do.
    def forecasts(qk_norm, scale):
        torch.manual_seed(0)
        model = KahlusPatchModel(ModelConfig(**SIZES["tiny"], patch_len=16, context=8, horizon=2, qk_norm=qk_norm))
        d = model.cfg.d_model
        with torch.no_grad():
            for block in model.blocks:
                block.qkv.bias[: 2 * d] = 0
                block.qkv.weight[: 2 * d] *= scale  # the query and key rows; the value rows stay
            torch.manual_seed(1)
            return model.predict(torch.randn(2, 8, 2, 16))[1]

    assert (forecasts(True, 1.0) - forecasts(True, 30.0)).abs().max() < 1e-3
    assert (forecasts(False, 1.0) - forecasts(False, 30.0)).abs().max() > 1e-2


def test_learning_rate_schedules():
    kw = {"lr": 1.0, "warmup": 10, "steps": 100}
    wsd = [train.lr_at(s, **kw) for s in range(100)]
    cosine = [train.lr_at(s, **kw, schedule="cosine") for s in range(100)]
    assert wsd[:10] == cosine[:10] == pytest.approx([0.1 * (s + 1) for s in range(10)])  # the same linear warm-up
    assert wsd[10:90] == [1.0] * 80  # stable at the peak until the last 10% of the run
    assert wsd[90:] == sorted(wsd[90:], reverse=True) and wsd[-1] == pytest.approx(0.1)
    assert all(a > b for a, b in zip(cosine[10:], cosine[11:])) and cosine[-1] == pytest.approx(0.1, abs=1e-3)  # falls at once


def test_batch_size_depends_on_data_size_only_and_stays_in_bounds():
    sizes = [1, 250, 10_000, 100_000, 1_000_000, 2_500_000, 10**7, 10**9]
    batches = [train.batch_for_data(n) for n in sizes]
    assert batches == sorted(batches)  # arXiv 2410.21676: more data, a batch that is not smaller
    assert all(64 <= b <= 4096 for b in batches) and batches[0] < batches[-1]
    assert list(inspect.signature(train.batch_for_data).parameters) == ["n_windows"]  # no model-size input


def test_defaults_tie_decay_to_lr_and_batch_to_data_and_warn_on_repetition(shard_dir, tmp_path, monkeypatch, capsys):
    for name in ("KAHLUS_MESA", "WORLD_SIZE", "RANK", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)
    base = ["--shards", str(shard_dir), "--size", "tiny", "--train-datasets", "synth_a", "--batch", "8", "--context", "8",
            "--horizon", "2", "--warmup", "1", "--workers", "0", "--lr", "6e-4"]
    train.main([*base, "--out", str(tmp_path / "short"), "--steps", "2"])
    state = torch.load(train.latest_checkpoint(tmp_path / "short"), weights_only=True)
    # The 10 fit nights hold 2,000 s, which is 250 windows of 8 s: the data rule needs more windows per step than one micro-batch of 8.
    ident = state["identity"]
    assert ident["accum"] * ident["batch"] * ident["world"] >= train.batch_for_data(250) and ident["accum"] > 1
    assert state["opt"]["param_groups"][0]["weight_decay"] == pytest.approx(1.5e-5 / 6e-4)
    assert "WARNING" not in capsys.readouterr().out  # 2 steps x 64 windows x 8 s = 0.5 passes over the data
    train.main([*base, "--out", str(tmp_path / "long"), "--steps", "20"])
    assert "reads the training data 5.1 times" in capsys.readouterr().out
    train.main([*base, "--out", str(tmp_path / "auto")])  # no --steps: derived from the data and --max-epochs 4
    ident = torch.load(train.latest_checkpoint(tmp_path / "auto"), weights_only=True)["identity"]
    seconds = ident["steps"] * ident["accum"] * ident["batch"] * ident["world"] * 8
    assert 3 <= seconds / 2000 <= 4 and "WARNING" not in capsys.readouterr().out  # at most 4 passes, and nearly 4


def test_run_directory_records_its_environment_and_scope(reference_run):
    (start,) = json.loads((reference_run / "environment.json").read_text())
    assert start["start_step"] == 0 and start["torch"] == torch.__version__ and start["gpu_count"] == torch.cuda.device_count()
    assert {"torch_cuda", "nccl", "gpu_names", "cuda_visible_devices", "world_size", "docker_image"} <= start.keys()
    text = run_env.summary(reference_run)
    assert "not a scientific claim" in text and f"1 to {STEPS}" in text
    assert (reference_run / "summary.md").read_text() == text

    run_env.main(["preflight", "--expect-gpus", str(torch.cuda.device_count())])
    with pytest.raises(SystemExit, match="preflight failed"):
        run_env.main(["preflight", "--expect-gpus", str(torch.cuda.device_count() + 1)])
