"""match_events' sorted-window search gives the full scan's matches, in the same order.

Every comparison is exact (==), on the matched timing errors and on the whole report.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from kahlus import montage_degrade
from kahlus.montage_degrade import _match_scan, _match_window, match_events


def _scan_report(detected, truth, tol_s):
    """match_events as it was before the window search: always the full scan."""
    real = montage_degrade._match_window
    montage_degrade._match_window = _match_scan
    try:
        return match_events(detected, truth, tol_s)
    finally:
        montage_degrade._match_window = real


def _same(a: dict, b: dict) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def _cohort(g, nights: int, *, per_night_true: int, per_night_det: int, grid: float | None = None):
    """Nights laid out on one time axis as score_nights does: each night's onsets, then a
    1e5 s gap. grid rounds times so that exact ties and gaps of exactly tol_s occur."""
    off, det, tru = 0.0, [], []
    for _ in range(nights):
        length = float(g.uniform(2e4, 3.5e4))
        t = g.uniform(0, length, int(g.integers(0, per_night_true + 1)))
        d = np.sort(g.uniform(0, length, int(g.integers(0, per_night_det + 1))))
        if grid:
            t, d = np.round(t / grid) * grid, np.round(d / grid) * grid
        tru.append(t + off)  # XML order: not always sorted
        det.append(d + off)
        off += length + 1e5
    return np.concatenate(det), np.concatenate(tru)


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("grid", [None, 0.5, 1.0])
def test_window_matches_scan_exactly(seed, grid):
    g = np.random.default_rng(seed)
    detected, truth = _cohort(g, 40, per_night_true=120, per_night_det=400, grid=grid)
    for tol_s in (3.0, 0.5, 1.0, 10.0):
        assert _match_window(detected, truth, tol_s) == _match_scan(detected, truth, tol_s)
        assert _same(match_events(detected, truth, tol_s), _scan_report(detected, truth, tol_s))


def test_dense_ties_and_duplicates():
    g = np.random.default_rng(5)
    for _ in range(200):
        truth = g.integers(0, 60, size=int(g.integers(0, 40))).astype(float)  # repeated onsets
        detected = g.integers(0, 60, size=int(g.integers(0, 60))).astype(float)
        if g.random() < 0.5:
            detected = np.sort(detected)
        for tol_s in (0.0, 1.0, 2.0, 3.0):
            assert _match_window(detected, truth, tol_s) == _match_scan(detected, truth, tol_s)
            assert _same(match_events(detected, truth, tol_s), _scan_report(detected, truth, tol_s))


def test_equidistant_tie_takes_the_lower_index():
    truth = np.array([12.0, 8.0, 12.0])
    assert _match_window(np.array([10.0]), truth, 3.0) == _match_scan(np.array([10.0]), truth, 3.0) == [-2.0]
    three = np.array([10.0, 10.0, 10.0])
    assert _match_window(three, truth, 3.0) == _match_scan(three, truth, 3.0) == [-2.0, 2.0, -2.0]


def test_gap_of_exactly_tol_matches_and_rounding_is_respected():
    truth = np.array([0.1 + 0.2, 7.3e8 + 0.25])  # 0.30000000000000004 and a large offset
    detected = np.array([3.3, 7.3e8 + 3.25, 7.3e8 - 2.75])
    for tol_s in (3.0, 2.9999999999999996, 3.0000000000000004):
        assert _match_window(detected, truth, tol_s) == _match_scan(detected, truth, tol_s)


def test_empty_inputs():
    e = np.array([])
    for detected, truth in ((e, e), (e, np.array([1.0])), (np.array([1.0]), e)):
        assert _same(match_events(detected, truth, 3.0), _scan_report(detected, truth, 3.0))


def test_non_finite_inputs_use_the_scan():
    truth = np.array([5.0, np.nan, 20.0])
    detected = np.array([5.5, 19.0, np.inf])
    assert _same(match_events(detected, truth, 3.0), _scan_report(detected, truth, 3.0))
    assert match_events(detected, truth, 3.0)["tp"] == 0  # a NaN onset wins every argmin
    truth = np.array([5.0, np.inf, 20.0])
    with np.errstate(invalid="ignore"):  # inf - inf in the scan
        assert _same(match_events(detected, truth, 3.0), _scan_report(detected, truth, 3.0))


def test_score_nights_report_is_unchanged(monkeypatch):
    """The montage study's scoring, run end to end with each matcher."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "run_montage_stress.py"
    spec = importlib.util.spec_from_file_location("run_montage_stress_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    g = np.random.default_rng(11)
    nights, probs = [], []
    for _ in range(30):
        t = np.arange(3.0, float(g.integers(2_000, 6_000)))
        p = np.clip(np.convolve(g.random(t.size), np.ones(5) / 5, mode="same") + g.normal(0, 0.1, t.size), 0, 1)
        nights.append({"t": t, "onsets": np.sort(g.uniform(t[0], t[-1], int(g.integers(0, 40))))})
        probs.append(p)
    for thr in np.linspace(0.05, 0.9, 18):
        window = mod.score_nights(nights, probs, thr)
        monkeypatch.setattr(montage_degrade, "_match_window", _match_scan)
        scan = mod.score_nights(nights, probs, thr)
        monkeypatch.undo()
        assert _same(window, scan)
