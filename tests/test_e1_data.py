"""The E1 quality metric flags flat, railed and burst patches, and nothing else."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

edfio = pytest.importorskip("edfio")
spec = importlib.util.spec_from_file_location("e1_data", Path(__file__).resolve().parents[1] / "scripts" / "e1_data.py")
e1_data = importlib.util.module_from_spec(spec)
sys.modules["e1_data"] = e1_data
spec.loader.exec_module(e1_data)


def test_flat_railed_and_burst_patches_are_the_bad_fraction(tmp_path):
    rng = np.random.default_rng(0)
    n = 180  # seconds, a whole number of 30 s epochs
    eeg = rng.normal(0, 20, n * 125)
    eeg[10 * 125: 30 * 125] = 0.0  # 20 flat seconds
    eeg[50 * 125 + 7] = 4990.0  # one railed second (physical max 5000)
    eeg[80 * 125: 81 * 125] *= 40  # one burst second (peak-to-peak over 10 times the median)
    ecg = rng.normal(0, 200, n * 125)
    edf = tmp_path / "shhs1-000001.edf"
    edfio.Edf([edfio.EdfSignal(eeg, 125, label="EEG", physical_range=(-5000, 5000)),
               edfio.EdfSignal(ecg, 125, label="ECG", physical_range=(-5000, 5000))]).write(edf)
    xml = tmp_path / "x.xml"
    xml.write_text("<PSGAnnotation><ScoredEvents><ScoredEvent><EventConcept>Stage 2 sleep|2</EventConcept><Start>0</Start><Duration>3600</Duration>"
                   "</ScoredEvent><ScoredEvent><EventConcept>Arousals|Arousals</EventConcept><Start>100</Start><Duration>5</Duration></ScoredEvent>"
                   "</ScoredEvents></PSGAnnotation>")
    subject, bad, ai, sleep_h = e1_data.one_night(("000001", edf, xml))
    assert bad == pytest.approx((20 + 1 + 1) / n, abs=0.01)
    assert ai == pytest.approx(1.0) and sleep_h == pytest.approx(1.0)
