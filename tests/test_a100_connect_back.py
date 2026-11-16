"""scripts/a100_connect_back.sh --dry-run: prints the plan, runs nothing, never echoes the key path."""
from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/a100_connect_back.sh"


def dry_run(tmp_path, *args, stdin=None):
    return subprocess.run(
        ["bash", str(SCRIPT), "--dry-run", *args],
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "USER": "krish"},
        input=stdin, capture_output=True, text=True, check=True, timeout=30,
    )


def test_key_path_never_echoed_and_nothing_runs(tmp_path):
    key = tmp_path / "very-secret-key-path.txt"
    key.write_text("tskey-secret")
    for mode in ([], ["--no-systemd"]):
        r = dry_run(tmp_path, "--auth-key-file", str(key), *mode)
        out = r.stdout + r.stderr
        assert "very-secret-key-path" not in out and "tskey-secret" not in out
        assert "--tailscale-hostname kahlus-a100" in out
        assert "--allow-anonymous" not in out and "--tailscale-allow-tailnet-members" not in out
    assert not (tmp_path / ".local").exists()  # dry run created no state
    assert not (tmp_path / ".config").exists()


def test_stdin_key_and_hostname_flag(tmp_path):
    r = dry_run(tmp_path, "--auth-key-file", "-", "--hostname", "a100-test", stdin="tskey-secret")
    assert "tskey-secret" not in r.stdout + r.stderr
    assert "https://a100-test.<tailnet>.ts.net" in r.stdout
