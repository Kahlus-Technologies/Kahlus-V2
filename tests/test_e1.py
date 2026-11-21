"""E1 pretraining additions, on CPU with the tiny model: the latent objective, its slow teacher, the ECG drop, night lists."""
import json
import shutil

import pytest

torch = pytest.importorskip("torch")

from kahlus import shards, train  # noqa: E402
from kahlus.model import SIZES, KahlusPatchModel, ModelConfig, future_targets  # noqa: E402


def _cfg(**kw):
    return ModelConfig(**SIZES["tiny"], patch_len=16, context=8, horizon=2, mlp_head=True, **kw)


def test_the_latent_teacher_gets_no_gradient_and_the_student_trains():
    torch.manual_seed(0)
    model = KahlusPatchModel(_cfg(objective="latent"))
    import copy

    teacher = copy.deepcopy(model).requires_grad_(False)
    window = torch.randn(4, 10, 2, 16)  # context 8 + horizon 2
    masked = torch.rand(4, 8, 2) < 0.5
    losses = model(window, masked, teacher.hidden(window))
    losses[0].backward()
    assert all(p.grad is None for p in teacher.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert float(model.last_target_std) > 0


def test_no_future_target_reads_a_second_at_or_before_its_token():
    seq = torch.arange(10.0).view(1, 10, 1, 1).expand(1, 10, 2, 3)  # second index as the value
    tgt = future_targets(seq, 8, 2)
    for s in range(8):
        assert tgt[0, s, 0, :, 0].tolist() == [s + 1.0, s + 2.0]


def test_ecg_tokens_of_dropped_windows_carry_no_loss():
    torch.manual_seed(0)
    model = KahlusPatchModel(_cfg())
    window = torch.randn(3, 10, 2, 16)
    masked = torch.rand(3, 8, 2) < 0.3
    masked[:, :, 1] = False
    masked[0, :, 1] = True  # window 0 drops its ECG
    other = window.clone()
    other[0, :, 1] += 5.0  # change only the dropped ECG
    with torch.no_grad():
        assert torch.allclose(model(window, masked), model(other, masked), atol=1e-6)
        other[1, :, 1] += 5.0  # a kept ECG does change the loss
        assert not torch.allclose(model(window, masked), model(other, masked), atol=1e-6)


@pytest.fixture(scope="module")
def shard_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("e1shards")
    shards.write_shards(shards.synthetic_nights(), out, shard_patches=1500)
    return out


ARGS = ["--size", "tiny", "--train-datasets", "synth_a", "--batch", "8", "--accum", "1", "--context", "8", "--horizon", "2", "--warmup", "10",
        "--log-every", "1", "--workers", "0", "--objective", "latent", "--head", "mlp", "--ecg-drop", "0.25", "--amp", "off"]


def _losses(out):
    return [json.loads(x) for x in (out / "log.jsonl").read_text().splitlines()]


def test_a_killed_latent_run_resumes_to_the_same_losses_and_the_target_does_not_collapse(shard_dir, tmp_path):
    full = tmp_path / "full"
    train.main(["--shards", str(shard_dir), "--out", str(full), "--steps", "50", "--ckpt-every", "25", *ARGS])
    assert _losses(full)[-1]["target_std"] > 0.1  # over 0.1 after 50 steps
    killed = tmp_path / "killed"  # a kill after step 25 leaves this checkpoint and 25 log lines
    killed.mkdir()
    shutil.copy(full / "ckpt-00000025.pt", killed / "ckpt-00000025.pt")
    (killed / "log.jsonl").write_text("".join(x + "\n" for x in (full / "log.jsonl").read_text().splitlines()[:25]))
    train.main(["--shards", str(shard_dir), "--out", str(killed), "--steps", "50", "--ckpt-every", "25", *ARGS])
    a, b = _losses(full), _losses(killed)
    assert max(abs(x["loss"] - y["loss"]) for x, y in zip(a, b)) < 1e-5 and len(a) == len(b)
    sa, sb = (torch.load(d / "ckpt-00000050.pt", weights_only=True) for d in (full, killed))
    assert all(torch.allclose(sa["teacher"][k], sb["teacher"][k], atol=1e-5) for k in sa["teacher"])


def test_a_night_list_with_a_held_out_or_dev_subject_is_refused(shard_dir, tmp_path):
    held = next(f"{i:06d}" for i in range(300000, 301000) if shards.is_heldout("shhs", f"{i:06d}"))
    for bad in (held, "200010"):
        f = tmp_path / "list.txt"
        f.write_text(bad + "\n")
        with pytest.raises(SystemExit, match="held-out or dev"):
            train.main(["--shards", str(shard_dir), "--out", str(tmp_path / "o"), "--steps", "2", "--night-list", str(f), *ARGS])
