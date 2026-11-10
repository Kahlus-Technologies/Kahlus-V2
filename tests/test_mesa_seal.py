"""MESA holdout seal: dev runs see only the 25 exposed nights; final sees only unseen ones."""
from pathlib import Path

import pytest

from kahlus import nsrr


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
    monkeypatch.setattr(nsrr, "MESA_FINAL_LOOKS", tmp_path / "looks")
    monkeypatch.setattr(nsrr, "MESA_FROZEN_PLAN", tmp_path / "untracked.md")  # not in git
    with pytest.raises(RuntimeError, match="committed"):
        nsrr.discover_nsrr_recordings(tmp_path, dataset="mesa")
    monkeypatch.setattr(nsrr, "_check_frozen", lambda: {"commit": "abc"})
    assert {r.subject_id for r in nsrr.discover_nsrr_recordings(tmp_path, dataset="mesa")} == {"9998", "9999"}
    assert not nsrr.mesa_fit_allowed()
    # same process may read again; a new process (different pid) is a second look and refuses
    assert len(nsrr.discover_nsrr_recordings(tmp_path, dataset="mesa")) == 2
    monkeypatch.setattr(nsrr.os, "getpid", lambda: -1)
    with pytest.raises(RuntimeError, match="already used"):
        nsrr.discover_nsrr_recordings(tmp_path, dataset="mesa")

    monkeypatch.setenv("KAHLUS_MESA", "all")
    with pytest.raises(ValueError):
        nsrr.discover_nsrr_recordings(tmp_path, dataset="mesa")


def test_shhs_untouched(tmp_path, monkeypatch):
    monkeypatch.delenv("KAHLUS_MESA", raising=False)
    for i in ("200001", "200002"):
        (tmp_path / f"shhs1-{i}.edf").write_bytes(b"")
        (tmp_path / f"shhs1-{i}-nsrr.xml").write_text("<x/>")
    assert len(nsrr.discover_nsrr_recordings(tmp_path, dataset="shhs")) == 2


def test_cache_and_census_paths_are_sealed(tmp_path, monkeypatch):
    """Readers that skip discover_nsrr_recordings (feature caches, annotation census) hit the same seal."""
    import sys

    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import nsrr_arousal_census
    import run_conformal_alarm

    from kahlus.nsrr_cache import CacheIdentityError, NightCache

    monkeypatch.delenv("KAHLUS_MESA", raising=False)
    exposed = sorted(nsrr.mesa_exposed_ids())[0]
    report = nsrr.DiscoveryReport(dataset="mesa", cohort="mesa", root=str(tmp_path), layout="split")
    cache = NightCache.bind(tmp_path / "cache", "mesa", producer="run_heart_arousal_benchmark",
                            report=report, ids=[exposed, "9999"])
    for i in (exposed, "9999"):
        cache.save(i, labels=np.zeros((3, 1)), x_prior=np.zeros((3, 2)))
    assert len(run_conformal_alarm.load(tmp_path / "cache", "mesa", 0)) == 1
    for i in (exposed, "9999"):  # a bare cache of unknown origin is refused outright
        np.savez(tmp_path / f"mesa-{i}.npz", labels=np.zeros((3, 1)), x_prior=np.zeros((3, 2)))
    with pytest.raises(CacheIdentityError):
        run_conformal_alarm.load(tmp_path, "mesa", 0)

    mesa = tmp_path / "mesa"
    mesa.mkdir()
    for i in (exposed, "9999"):
        (mesa / f"mesa-sleep-{i}-nsrr.xml").write_text("<PSGAnnotation><ScoredEvents/></PSGAnnotation>")
    monkeypatch.setattr(sys, "argv", ["census", str(mesa), "--out", str(tmp_path / "out")])
    nsrr_arousal_census.main()
    assert len((tmp_path / "out/mesa_per_subject.jsonl").read_text().splitlines()) == 1
