"""Feature caches carry identity: data root, ID list, producer and commit, or they are refused."""
from __future__ import annotations

import numpy as np
import pytest

from kahlus import nsrr_cache
from kahlus.nsrr import DiscoveryReport
from kahlus.nsrr_cache import IDENTITY_KEY, CacheIdentityError, NightCache


def _report(root: str = "/data/shhs") -> DiscoveryReport:
    return DiscoveryReport(dataset="shhs", cohort="shhs", root=root, layout="split",
                           edf_dir=f"{root}/edfs", annotation_dir=f"{root}/annotations")


@pytest.fixture(autouse=True)
def _dev(monkeypatch):
    monkeypatch.delenv("KAHLUS_MESA", raising=False)


def _bind(tmp_path, ids=("1", "2"), root="/data/shhs", producer="run_heart_arousal_benchmark"):
    return NightCache.bind(tmp_path, "shhs", producer=producer, report=_report(root), ids=ids)


def test_roundtrip_strips_identity_and_rebinds(tmp_path):
    c = _bind(tmp_path)
    c.save("1", labels=np.ones(3))
    assert set(np.load(tmp_path / "shhs-1.npz").files) == {"labels", IDENTITY_KEY}
    assert list(c.load("1")) == ["labels"]
    again = _bind(tmp_path)  # same run identity reopens it (diagnostic reruns in one run dir)
    assert again.has("1") and not again.has("2")
    assert NightCache.open(tmp_path, "shhs", producer="run_heart_arousal_benchmark").subjects() == ["1"]


@pytest.mark.parametrize("change", [{"ids": ("1", "3")}, {"root": "/other/shhs"}, {"producer": "run_beat_warning"}])
def test_a_different_run_is_refused(tmp_path, change):
    _bind(tmp_path)
    with pytest.raises(CacheIdentityError, match="different run"):
        _bind(tmp_path, **change)


def test_cache_without_identity_is_refused(tmp_path):
    np.savez(tmp_path / "shhs-1.npz", labels=np.ones(3))  # an old cache from before identity
    with pytest.raises(CacheIdentityError, match="unknown origin"):
        _bind(tmp_path)
    with pytest.raises(CacheIdentityError, match="unknown origin"):
        NightCache.open(tmp_path, "shhs", producer="run_heart_arousal_benchmark")


def test_entries_without_matching_identity_are_refused(tmp_path):
    c = _bind(tmp_path)
    np.savez(tmp_path / "shhs-1.npz", labels=np.ones(3))  # no identity key
    with pytest.raises(CacheIdentityError, match="lacks"):
        c.load("1")
    other = NightCache.bind(tmp_path / "other", "shhs", producer="x", report=_report(), ids=["2"])
    other.save("2", labels=np.ones(3))
    (tmp_path / "other/shhs-2.npz").rename(tmp_path / "shhs-2.npz")  # a night copied from another run
    with pytest.raises(CacheIdentityError, match="lacks"):
        c.load("2")
    with pytest.raises(CacheIdentityError, match="not on"):
        c.save("9", labels=np.ones(3))


def test_reader_refuses_another_commit_or_producer(tmp_path, monkeypatch):
    _bind(tmp_path).save("1", labels=np.ones(3))
    with pytest.raises(CacheIdentityError, match="producer"):
        NightCache.open(tmp_path, "shhs", producer="run_beat_warning")
    real = nsrr_cache.git_state()
    monkeypatch.setattr(nsrr_cache, "git_state", lambda: real | {"commit": "0" * 40})
    with pytest.raises(CacheIdentityError, match="commit"):
        NightCache.open(tmp_path, "shhs", producer="run_heart_arousal_benchmark")
