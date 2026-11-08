"""MESA holdout seal: dev runs see only the 25 exposed nights; final sees only unseen ones."""
from pathlib import Path

import pytest

from neurotwin.adapters import nsrr


def _fake(root: Path, ids: list[str]) -> None:
    for i in ids:
        (root / f"mesa-sleep-{i}.edf").write_bytes(b"")
        (root / f"mesa-sleep-{i}-nsrr.xml").write_text("<x/>")


def test_mesa_seal(tmp_path, monkeypatch):
    exposed = sorted(nsrr.mesa_exposed_ids())
    assert len(exposed) == 25
    _fake(tmp_path, exposed[:2] + ["9998", "9999"])

    monkeypatch.delenv("KAHLUS_MESA", raising=False)
    assert {r.subject_id for r in nsrr.discover_nsrr_recordings(tmp_path, dataset="mesa")} == set(exposed[:2])

    monkeypatch.setenv("KAHLUS_MESA", "final")
    monkeypatch.setattr(nsrr, "MESA_FROZEN_PLAN", tmp_path / "missing.md")
    with pytest.raises(RuntimeError):
        nsrr.discover_nsrr_recordings(tmp_path, dataset="mesa")
    plan = tmp_path / "plan.md"
    plan.write_text("frozen")
    monkeypatch.setattr(nsrr, "MESA_FROZEN_PLAN", plan)
    assert {r.subject_id for r in nsrr.discover_nsrr_recordings(tmp_path, dataset="mesa")} == {"9998", "9999"}

    monkeypatch.setenv("KAHLUS_MESA", "all")
    with pytest.raises(ValueError):
        nsrr.discover_nsrr_recordings(tmp_path, dataset="mesa")


def test_shhs_untouched(tmp_path, monkeypatch):
    monkeypatch.delenv("KAHLUS_MESA", raising=False)
    for i in ("200001", "200002"):
        (tmp_path / f"shhs1-{i}.edf").write_bytes(b"")
        (tmp_path / f"shhs1-{i}-nsrr.xml").write_text("<x/>")
    assert len(nsrr.discover_nsrr_recordings(tmp_path, dataset="shhs")) == 2
