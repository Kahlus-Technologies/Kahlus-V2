from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from neurotwin.adapters.nsrr import (
    discover_nsrr_recordings,
    epoch_arousal_mask,
    load_nsrr_epoch_matrix,
    parse_nsrr_arousal_events,
)
from neurotwin.forecastability.autonomic_rfs import (
    load_nsrr_autonomic_fixture,
    make_autonomic_fixture,
    run_autonomic_rfs_gate,
)


class NsrrAdapterTests(unittest.TestCase):
    def test_parse_arousal_events_from_xml(self) -> None:
        xml = """<?xml version="1.0"?>
        <PSGAnnotation>
          <ScoredEvents>
            <ScoredEvent>
              <Start>30.0</Start>
              <Duration>5.0</Duration>
              <EventConcept>Arousal ()</EventConcept>
            </ScoredEvent>
            <ScoredEvent>
              <Start>120.0</Start>
              <Duration>3.0</Duration>
              <EventConcept>Stage 2 sleep</EventConcept>
            </ScoredEvent>
          </ScoredEvents>
        </PSGAnnotation>
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test-nsrr.xml"
            path.write_text(xml, encoding="utf-8")
            events = parse_nsrr_arousal_events(path)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0][0], 30.0)
        self.assertAlmostEqual(events[0][1], 5.0)

    def test_epoch_arousal_mask(self) -> None:
        mask = epoch_arousal_mask([(25.0, 10.0)], n_epochs=4, epoch_seconds=30.0)
        self.assertEqual(int(mask[0]), 1)
        self.assertEqual(int(mask[1]), 1)
        self.assertEqual(int(mask[2]), 0)

    def test_load_nsrr_epoch_matrix_channel_axis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            edf = self._write_stub_edf(root / "mesa-sleep-0001.edf")
            try:
                matrix = load_nsrr_epoch_matrix(edf, epoch_seconds=30.0)
            except RuntimeError as exc:
                if "mne" in str(exc).lower():
                    self.skipTest("mne not installed")
                raise
            self.assertIn("eeg", matrix["epochs"])
            eeg = matrix["epochs"]["eeg"]
            self.assertEqual(eeg.ndim, 3)
            self.assertEqual(eeg.shape[0], matrix["n_epochs"])
            self.assertGreaterEqual(eeg.shape[2], 1)
            self.assertIsInstance(matrix["channels"]["eeg"], list)

    def test_load_nsrr_autonomic_fixture_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            try:
                self._write_stub_nsrr_cohort(root, n_subjects=4)
                fixture = load_nsrr_autonomic_fixture(
                    root,
                    dataset_code=0,
                    dataset_name="mesa",
                    horizons=(1, 2),
                )
            except RuntimeError as exc:
                if "mne" in str(exc).lower():
                    self.skipTest("mne not installed")
                raise
            self.assertEqual(fixture.eeg_epochs.ndim, 3)
            self.assertEqual(fixture.autonomic_epochs.ndim, 3)
            self.assertEqual(fixture.eeg_epochs.shape[1], 128)
            self.assertGreaterEqual(fixture.autonomic_epochs.shape[2], 1)
            self.assertGreaterEqual(len(set(fixture.subject.tolist())), 3)
            self.assertGreater(int(np.sum(fixture.y_by_horizon[1])), 0)
            self.assertIsNotNone(fixture.channels_used)

    def test_autonomic_gate_evaluates_stub_mesa(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = Path(tmp) / "out"
            try:
                self._write_stub_nsrr_cohort(root, n_subjects=4)
                gate = run_autonomic_rfs_gate(
                    out,
                    seed=2,
                    mesa_root=root,
                    shhs_root=None,
                    bootstrap_mode="smoke",
                    max_recordings=4,
                    min_subjects=3,
                )
            except RuntimeError as exc:
                if "mne" in str(exc).lower():
                    self.skipTest("mne not installed")
                raise
            self.assertEqual(gate["mesa_status"], "evaluated")
            self.assertIsNotNone(gate["mesa_real"])
            self.assertGreaterEqual(gate["mesa_real"]["n_subjects"], 3)

    def test_fixture_has_multiple_subjects(self) -> None:
        fixture = make_autonomic_fixture(seed=2, residual_signal=True)
        self.assertGreaterEqual(len(set(fixture.subject.tolist())), 8)

    def test_discover_requires_xml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "orphan.edf").write_bytes(b"")
            self.assertEqual(discover_nsrr_recordings(root, dataset="mesa"), [])

    def _write_stub_nsrr_cohort(self, root: Path, *, n_subjects: int) -> None:
        for i in range(1, n_subjects + 1):
            stem = f"mesa-sleep-{i:04d}"
            edf = root / f"{stem}.edf"
            xml = root / f"{stem}-nsrr.xml"
            self._write_stub_edf(edf, seed=i)
            xml.write_text(
                f"""<?xml version="1.0"?>
                <PSGAnnotation>
                  <ScoredEvents>
                    <ScoredEvent>
                      <Start>{30.0 * (i % 3)}</Start>
                      <Duration>8.0</Duration>
                      <EventConcept>Arousal ()</EventConcept>
                    </ScoredEvent>
                    <ScoredEvent>
                      <Start>{90.0 + 10 * i}</Start>
                      <Duration>6.0</Duration>
                      <EventConcept>Arousal|Arousal ()</EventConcept>
                    </ScoredEvent>
                  </ScoredEvents>
                </PSGAnnotation>
                """,
                encoding="utf-8",
            )

    def _write_stub_edf(self, path: Path, *, seed: int = 0) -> Path:
        import mne

        sfreq = 100.0
        # 5 minutes -> enough 30s epochs for horizons
        n_times = int(sfreq * 300)
        rng = np.random.default_rng(seed)
        t = np.arange(n_times) / sfreq
        ch_names = ["EEG C3-A2", "EEG C4-A1", "ECG", "EOG E1-A2", "EMG Chin", "AIRFLOW"]
        data = rng.normal(0.0, 0.05, size=(len(ch_names), n_times))
        data[0] += 0.3 * np.sin(2 * np.pi * 3 * t)
        data[2] += 0.2 * np.sin(2 * np.pi * 1.2 * t)
        info = mne.create_info(
            ch_names,
            sfreq,
            ch_types=["eeg", "eeg", "ecg", "eog", "emg", "misc"],
        )
        raw = mne.io.RawArray(data, info, verbose=False)
        mne.export.export_raw(path, raw, fmt="edf", overwrite=True, verbose=False)
        return path


if __name__ == "__main__":
    unittest.main()
