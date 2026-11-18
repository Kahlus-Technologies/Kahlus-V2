"""The A100 handoff: the evidence zip never takes secrets, checkpoints or data, and the verify stage
catches a changed file. The package test packs HEAD, so it checks committed scripts only."""
from __future__ import annotations

import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TOKEN = "nsrr-token-value-0123456789"


def _write(path: Path, text: str = "{}") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_evidence_zip_keeps_records_and_drops_secrets_checkpoints_and_data(tmp_path):
    home, work = tmp_path / "home", tmp_path / "work"
    _write(home / ".nsrr_token", TOKEN + "\n")
    keep = ["verify.txt", "preflight/environment.json", "logs/train10m.log", "runs/kahlus-10m/summary.md",
            "runs/kahlus-10m/log.jsonl", "runs/kahlus-10m/benchmark.json", "devslice/r/steps.tsv",
            "devslice/r/discovery/ids_shhs.txt"]
    drop_files = {
        "logs/.env": "secret", "logs/config.env": "secret", "logs/.nsrr_token": "secret", "runs/api_token.txt": "secret",
        "logs/id_ed25519": "secret", "logs/server.key": "secret", "logs/cert.pem": "secret", "logs/pw.txt": "secret",
        "logs/ts-key.txt": "secret", "logs/accessKeys.csv": "secret",
        "runs/kahlus-10m/ckpt-00000020.pt": "checkpoint", "runs/kahlus-10m/ckpt-00000040.pt.tmp": "checkpoint",
        "devslice/r/census/mesa_per_subject.jsonl": "data", "runs/night.edf": "data", "logs/shhs1-dataset-0.21.0.csv": "data",
        "logs/fetch.log": "holds the NSRR token", "logs/ssh.log": "holds a secret", "logs/env.log": "holds a secret",
        "logs/hf.log": "holds a secret", "logs/bearer.log": "holds a secret", "runs/kaggle.json": "holds a secret",
        "runs/kahlus-10m/curve.png": "not a text record", "logs/bad\nname.log": "odd name",
    }
    drop_dirs = ["runs/venv-host", "smoke/shards", "devslice/r/bench/cache"]
    stray = ["ts-key.txt", ".env", "venv-host", "nsrr", "NSRR-data", "shards", "home"]  # in WORK, outside the run outputs
    for rel in [*keep, *drop_files, *stray]:
        _write(work / rel)
    for rel in drop_dirs:
        _write(work / rel / "inside.json")
    _write(work / "logs/fetch.log", f"GET https://sleepdata.org/?auth_token={TOKEN}\n")
    _write(work / "logs/ssh.log", "-----BEGIN OPENSSH PRIVATE KEY-----\nabc\n")
    _write(work / "logs/env.log", "WANDB_API_KEY=0123456789abcdef\n")
    _write(work / "logs/hf.log", "login with hf_abcdefghijklmnopqrstuvwxyz0123\n")
    _write(work / "logs/bearer.log", "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.abcdef\n")
    _write(work / "runs/kaggle.json", '{"username": "k", "key": "0123456789abcdef0123"}\n')
    _write(work / "logs/bad\ndir/a.log")
    (work / "logs/token-link.json").symlink_to(home / ".nsrr_token")
    os.mkfifo(work / "logs/stuck.log")

    out = work / "kahlus-evidence-abc.zip"
    r = subprocess.run([str(REPO / "scripts/package_evidence.sh"), str(work), str(out)],
                       env=os.environ | {"HOME": str(home)}, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr

    with zipfile.ZipFile(out) as z:
        names = {n.split("/", 1)[1] for n in z.namelist()}
        excluded = dict(line.split("\t") for line in z.read("kahlus-evidence-abc/EXCLUDED.txt").decode().splitlines())
        manifest = z.read("kahlus-evidence-abc/MANIFEST.sha256").decode()
        assert all(TOKEN.encode() not in z.read(n) for n in z.namelist())
    assert names == set(keep) | {"MANIFEST.sha256", "EXCLUDED.txt"}
    assert excluded == ({k.replace("\n", "\\n"): v for k, v in drop_files.items()} | dict.fromkeys(drop_dirs, "skipped directory")
                        | dict.fromkeys(stray, "not a run output")
                        | {"logs/token-link.json": "symlink", "logs/stuck.log": "not a regular file", "logs/bad\\ndir": "odd name"})
    assert sorted(line.split("  ")[1] for line in manifest.splitlines()) == sorted(keep)


@pytest.mark.skipif(not (shutil.which("git") and shutil.which("sha256sum")), reason="needs git and sha256sum")
def test_handoff_package_verifies_and_the_verify_stage_catches_a_changed_file(tmp_path):
    r = subprocess.run([str(REPO / "scripts/make_handoff.sh"), "HEAD", str(tmp_path / "dist")],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    name, tarball = f"kahlus-a100-handoff-{sha[:12]}", f"kahlus-v2-{sha[:12]}.tar.gz"
    with zipfile.ZipFile(tmp_path / "dist" / f"{name}.zip") as z:
        assert {n for n in z.namelist() if not n.endswith("/")} == {
            f"{name}/{f}" for f in ("COMMIT_HASH.txt", "README_HANDOFF.md", "SHA256SUMS", tarball)}
        z.extractall(tmp_path)
    pkg = tmp_path / name
    assert (pkg / "COMMIT_HASH.txt").read_text().strip() == sha
    subprocess.run(["sha256sum", "--check", "--strict", "SHA256SUMS"], cwd=pkg, check=True, capture_output=True)
    subprocess.run(["tar", "xzf", tarball], cwd=pkg, check=True)

    def verify():
        return subprocess.run([str(pkg / "kahlus-v2/scripts/a100_handoff.sh"), "verify"],
                              env=os.environ | {"WORK": str(tmp_path / "work")}, capture_output=True, text=True, timeout=60)

    ok = verify()
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert f"commit {sha}" in (tmp_path / "work/verify.txt").read_text()

    train = pkg / "kahlus-v2/src/kahlus/train.py"
    original = train.read_bytes()
    train.write_text("# changed\n")
    bad = verify()
    assert bad.returncode != 0 and "HANDOFF_SHA256SUMS" in bad.stdout + bad.stderr
    train.write_bytes(original)
    (pkg / tarball).unlink()  # deleting the tarball after unpacking is allowed
    assert verify().returncode == 0

    stray = pkg / "kahlus-v2/src/sklearn.py"  # an added file would shadow a library
    stray.write_text("raise SystemExit\n")
    bad = verify()
    assert bad.returncode != 0 and "src/sklearn.py" in bad.stdout + bad.stderr
    stray.unlink()

    (pkg / "README_HANDOFF.md").write_text("changed\n")
    bad = verify()
    assert bad.returncode != 0 and f"{pkg}/SHA256SUMS" in bad.stdout + bad.stderr
