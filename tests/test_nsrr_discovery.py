"""NSRR discovery: split and co-located layouts, counted rejects, root and ID overrides, MESA seal."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from kahlus import nsrr

REPO = Path(__file__).resolve().parents[1]


def _touch(path: Path, text: str = "<PSGAnnotation><ScoredEvents/></PSGAnnotation>") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("KAHLUS_MESA", "NSRR_ROOT", "NSRR_MESA_ROOT", "NSRR_SHHS_ROOT", "NSRR_MESA_IDS", "NSRR_SHHS_IDS"):
        monkeypatch.delenv(k, raising=False)


def test_split_layout_pairs_by_stem_and_counts_rejects(tmp_path):
    root = tmp_path / "raw"
    for i in ("200001", "200002"):
        _touch(root / "edfs" / f"shhs1-{i}.edf")
        _touch(root / "annotations" / f"shhs1-{i}-nsrr.xml")
    _touch(root / "edfs" / "shhs1-200003.edf")  # no annotation
    _touch(root / "annotations" / "shhs1-200004-nsrr.xml")  # no EDF
    _touch(root / "annotations" / "shhs1-200001-profusion.xml")  # other schema, never read
    recs, rep = nsrr.discover_nsrr(root, dataset="raw", quiet=True)
    assert [r.subject_id for r in recs] == ["200001", "200002"]
    assert recs[0].xml_path == root / "annotations" / "shhs1-200001-nsrr.xml"
    assert (rep.layout, rep.cohort, rep.n_edf, rep.n_xml) == ("split", "shhs", 3, 3)
    assert (rep.unmatched_edf, rep.unmatched_xml, rep.n_xml_ignored, rep.n_recordings) == (1, 1, 1, 2)
    assert rep.examples == {"unmatched_edf": ["shhs1-200003"], "unmatched_xml": ["shhs1-200004"]}
    assert rep.recordings_sha256 == nsrr.ids_sha256(["200001", "200002"])


def test_split_layout_with_nsrr_tree_names_and_subfolders(tmp_path):
    _touch(tmp_path / "edfs" / "shhs1" / "shhs1-200001.edf")
    _touch(tmp_path / "annotations-events-nsrr" / "shhs1" / "shhs1-200001-nsrr.xml")
    recs, rep = nsrr.discover_nsrr(tmp_path, dataset="shhs", quiet=True)
    assert len(recs) == 1 and rep.layout == "split"


def test_colocated_layout_keeps_old_behavior_and_counts(tmp_path):
    _touch(tmp_path / "shhs1-200001.edf")
    _touch(tmp_path / "shhs1-200001-nsrr.xml")
    _touch(tmp_path / "shhs1-200002.edf")
    _touch(tmp_path / "shhs1-200002.xml")  # plain .xml still pairs
    _touch(tmp_path / "a" / "shhs1-200003.edf")
    _touch(tmp_path / "b" / "shhs1-200003-nsrr.xml")  # different folder: not a pair
    recs, rep = nsrr.discover_nsrr(tmp_path, dataset="shhs", quiet=True)
    assert [r.subject_id for r in recs] == ["200001", "200002"]
    assert (rep.layout, rep.unmatched_edf, rep.unmatched_xml) == ("co-located", 1, 1)
    assert nsrr.discover_nsrr_recordings(tmp_path, dataset="shhs") == recs


def test_duplicate_stems_and_subject_ids_are_dropped(tmp_path):
    _touch(tmp_path / "edfs" / "x" / "shhs1-200001.edf")
    _touch(tmp_path / "edfs" / "y" / "shhs1-200001.edf")
    _touch(tmp_path / "annotations" / "shhs1-200001-nsrr.xml")
    for stem in ("shhs1-200002", "shhs2-200002"):  # same subject ID twice
        _touch(tmp_path / "edfs" / f"{stem}.edf")
        _touch(tmp_path / "annotations" / f"{stem}-nsrr.xml")
    _touch(tmp_path / "edfs" / "shhs1-200005.edf")
    _touch(tmp_path / "annotations" / "shhs1-200005-nsrr.xml")
    recs, rep = nsrr.discover_nsrr(tmp_path, dataset="shhs", quiet=True)
    assert [r.subject_id for r in recs] == ["200005"]
    assert rep.duplicate == 5 and rep.n_matched == 1


def test_root_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert nsrr.nsrr_root("shhs") == tmp_path / "home/data/nsrr/shhs"
    monkeypatch.setenv("NSRR_ROOT", str(tmp_path / "nsrr"))
    assert nsrr.nsrr_root("shhs") == tmp_path / "nsrr/shhs"
    monkeypatch.setenv("NSRR_SHHS_ROOT", str(tmp_path / "raw"))
    assert nsrr.nsrr_root("shhs") == tmp_path / "raw"
    assert nsrr.nsrr_root("mesa") == tmp_path / "nsrr/mesa"


def test_id_filter_from_env(tmp_path, monkeypatch):
    for i in ("200001", "200002", "200003"):
        _touch(tmp_path / "edfs" / f"shhs1-{i}.edf")
        _touch(tmp_path / "annotations" / f"shhs1-{i}-nsrr.xml")
    ids = tmp_path / "ids.txt"
    ids.write_text("# subset\n200001\nshhs1-200003\n200009\n")
    monkeypatch.setenv("NSRR_SHHS_IDS", str(ids))
    recs, rep = nsrr.discover_nsrr(tmp_path, dataset="raw", quiet=True)
    assert [r.subject_id for r in recs] == ["200001", "200003"]
    assert (rep.n_id_filter, rep.ids_not_found, rep.n_id_filtered_out) == (3, 1, 1)
    assert rep.id_filter_sha256 == nsrr.ids_sha256(["200001", "200003", "200009"])


def test_mesa_seal_holds_in_split_layout_and_against_id_filter(tmp_path, monkeypatch):
    exposed = sorted(nsrr.mesa_exposed_ids())[:2]
    for i in [*exposed, "9998", "9999"]:
        _touch(tmp_path / "edfs" / f"mesa-sleep-{i}.edf")
        _touch(tmp_path / "annotations" / f"mesa-sleep-{i}-nsrr.xml")
    _touch(tmp_path / "edfs" / "mesa-sleep-9997.edf")
    recs, rep = nsrr.discover_nsrr(tmp_path, dataset="raw_mesa", quiet=True)  # folder name says nothing
    assert {r.subject_id for r in recs} == set(exposed)
    assert (rep.cohort, rep.n_mesa_gated_out, rep.unmatched_edf) == ("mesa", 2, 1)
    assert rep.examples == {}  # MESA stems never appear in reports
    ids = tmp_path / "ids.txt"
    ids.write_text(f"9999\n{exposed[0]}\n")
    monkeypatch.setenv("NSRR_MESA_IDS", str(ids))  # a list cannot widen the seal
    assert [r.subject_id for r in nsrr.discover_nsrr_recordings(tmp_path, dataset="raw_mesa")] == [exposed[0]]
    xmls, arep = nsrr.discover_nsrr_annotations(tmp_path, dataset="raw_mesa")
    assert [s for s, _ in xmls] == [exposed[0]] and arep.n_mesa_gated_out == 1 and arep.n_id_filtered_out == 2


def test_annotations_count_without_edf(tmp_path):
    _touch(tmp_path / "edfs" / "shhs1-200001.edf")
    _touch(tmp_path / "annotations" / "shhs1-200001-nsrr.xml")
    _touch(tmp_path / "annotations" / "shhs1-200002-nsrr.xml")
    xmls, rep = nsrr.discover_nsrr_annotations(tmp_path, dataset="shhs")
    assert [s for s, _ in xmls] == ["200001", "200002"] and rep.layout == "split"


def test_missing_root_reports_instead_of_raising(tmp_path):
    recs, rep = nsrr.discover_nsrr(tmp_path / "nope", dataset="shhs", quiet=True)
    assert recs == [] and rep.layout == "missing"


def _discover_cli(*args: str) -> subprocess.CompletedProcess:
    env = {"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin", "HOME": "/nonexistent"}
    return subprocess.run([sys.executable, str(REPO / "scripts/nsrr_discover.py"), *args],
                          capture_output=True, text=True, env=env)


def test_subset_is_deterministic_and_never_overwritten(tmp_path):
    for i in range(200001, 200021):
        _touch(tmp_path / "raw/edfs" / f"shhs1-{i}.edf")
        _touch(tmp_path / "raw/annotations" / f"shhs1-{i}-nsrr.xml")
    runs = []
    for out in ("a", "b"):
        r = _discover_cli("--cohort", "shhs", "--root", str(tmp_path / "raw"), "--out", str(tmp_path / out), "--limit", "5")
        assert r.returncode == 0, r.stderr
        runs.append(nsrr.read_id_list(tmp_path / out / "ids_shhs.txt"))
    assert runs[0] == runs[1] and len(runs[0]) == 5
    r = _discover_cli("--cohort", "shhs", "--root", str(tmp_path / "raw"), "--out", str(tmp_path / "a"), "--limit", "5")
    assert r.returncode != 0  # the subset file of a run is never replaced
    r = _discover_cli("--cohort", "shhs", "--root", str(tmp_path / "empty"), "--out", str(tmp_path / "c"))
    assert r.returncode == 1 and (tmp_path / "c/discovery_shhs.json").exists()
