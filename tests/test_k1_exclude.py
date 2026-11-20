from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_heart_arousal_benchmark as bench  # noqa: E402

from kahlus.heart_arousal import anchor_labels, build_recording  # noqa: E402
from kahlus.nsrr import read_id_list  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def test_dev_ids_file_lists_the_25_dev_nights() -> None:
    assert read_id_list(REPO / "configs/data/shhs_dev_ids.txt") == [str(i) for i in range(200001, 200026)]


def test_excluded_ids_never_reach_feature_building(tmp_path, monkeypatch) -> None:
    ids = [str(i) for i in range(200020, 200030)]
    recs = [SimpleNamespace(subject_id=i, edf_path=f"{i}.edf", xml_path=f"{i}.xml") for i in ids]
    report = SimpleNamespace(root="/x", layout="co-located", edf_dir="/x", annotation_dir="/x", to_json=lambda: {})
    built = []

    def fake_build(edf, xml, **kw):
        built.append(Path(edf).stem)
        return {"anchors": np.arange(3.0), "labels": np.zeros((3, 5), dtype=np.int8),
                "blocks": {"cardiac": np.zeros((3, 2)), "prior": np.zeros((3, 2)), "eeg": np.zeros((3, 2))}}

    monkeypatch.setattr(bench, "discover_nsrr", lambda root, dataset: (recs, report))
    monkeypatch.setattr(bench, "build_recording", fake_build)
    dev = frozenset(read_id_list(REPO / "configs/data/shhs_dev_ids.txt"))
    out = bench.load_cohort("shhs", Path("/x"), tmp_path, None, "causal", dev)
    assert sorted(built) == ["200026", "200027", "200028", "200029"]
    assert out["n_nights"] == 4


def test_arousal_lag_moves_only_arousal_onsets(monkeypatch) -> None:
    seen = {}
    import kahlus.heart_arousal as ha

    events = [("Arousal ()", 100.0, 5.0), ("Stage 2 sleep", 90.0, 30.0)]
    monkeypatch.setattr("kahlus.nsrr.parse_nsrr_events", lambda p: events)
    monkeypatch.setattr("kahlus.nsrr.load_nsrr_epoch_matrix", lambda *a, **k: {"sfreq": 1.0, "epochs": {"ecg": np.zeros((1, 400, 1)), "eeg": np.zeros((1, 400, 1))}})
    monkeypatch.setattr("kahlus.qrs_fast.causal_beats", lambda x, fs: (np.zeros(0), np.zeros(0)))
    monkeypatch.setattr(ha, "anchor_labels", lambda ev, **k: seen.setdefault("ev", ev) and (np.zeros(0), np.zeros((0, 5), dtype=np.int8)))
    monkeypatch.setattr(ha, "cardiac_features", lambda *a: np.zeros((0, 1)))
    monkeypatch.setattr(ha, "prior_features", lambda *a: np.zeros((0, 1)))
    monkeypatch.setattr(ha, "eeg_features", lambda *a, **k: np.zeros((0, 1)))
    build_recording("e", "x", ecg_labels=("a",), eeg_labels=("b",), arousal_lag_s=2.0)
    assert seen["ev"] == [("Arousal ()", 98.0, 5.0), ("Stage 2 sleep", 90.0, 30.0)]
